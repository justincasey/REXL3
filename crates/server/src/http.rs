//! The HTTP surface: chat completions, plain completions, a render endpoint for exact prompt tokens, models,
//! health and Prometheus-style metrics. Streaming uses server-sent events.

use crate::api::{self, ApiError, GenDefaults, Params};
use crate::chat_format::ChatFormat;
use crate::flow::{channel_with_failure, FailureHook, Message, Receiver};
use crate::observability::{Arrival, SseOutcome};
use crate::reasoning::{ReasoningCount, ReasoningSplitter};
use crate::runtime::{CancelGuard, DoneInfo, EngineHandle, FailKind, GenRequest, Refusal, SeqError, SeqMsg};
use crate::template::TemplateError;
use crate::tokenizer::{StreamDecoder, Tok};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Extension;
use axum::{Json, Router};
use futures_util::StreamExt;
use gb_engine::SubmitError;
use gb_sched::AdmitError;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct AppState {
    pub handle: EngineHandle,
    pub tok: Arc<Tok>,
    /// The model family's chat conventions: template, thinking markers, tool-call syntax, image placeholders.
    pub format: Arc<dyn ChatFormat>,
    pub defaults: GenDefaults,
    pub model_name: String,
    pub model_aliases: Vec<String>,
    pub request_budget: Arc<crate::budget::RequestBudget>,
    pub max_model_len: usize,
    /// How often a streaming reply sends an SSE comment (`: keep-alive`) while it has nothing else to say, so a client and the proxies
    /// between us see life while the request is queued or its prompt is being prefilled.
    pub sse_keepalive: Duration,
    /// New tokens a request may generate when it names no `max_tokens` (0: whatever fits in the context window).
    /// Admission reserves KV for the whole budget up front, so an unbounded default lets one client hold the entire
    /// pool and makes every other default client wait behind it.
    pub default_max_tokens: usize,
    /// Image input: the encoder (published once the tower is loaded) and the request limits; `None` refuses image parts.
    pub vision: Option<VisionHandle>,
    /// What a chat request that says nothing about thinking gets (`--default-thinking`; `off` unless the flag says otherwise).
    pub default_thinking: crate::reasoning_controls::DefaultThinking,
    /// The model's reasoning capabilities, checked against its template and tokenizer at load (`ReasoningSupport::for_format`): every
    /// reasoning decision of a request reads this, and `/v1/models` advertises it.
    pub reasoning_caps: Arc<crate::reasoning_caps::ReasoningSupport>,
}

impl AppState {
    /// The kwargs and the reasoning budget of a chat request, under this family's controls and the server's default.
    pub fn reasoning(&self, body: &Value) -> Result<crate::reasoning_controls::Resolved, ApiError> {
        crate::reasoning_controls::resolve(body, &self.reasoning_caps.caps, self.default_thinking)
    }

    /// The top-level reasoning fields the field allow-list passes for this model.
    pub fn reasoning_fields(&self) -> Vec<String> {
        crate::reasoning_controls::known_top_level(&self.reasoning_caps.caps)
    }
}

#[derive(Clone)]
pub struct VisionHandle {
    pub slot: gb_model::serve::VisionSlot,
    pub limits: crate::vision::Limits,
    /// The family's image preprocessing (decode, canvas and token count, patch layout).
    pub pp: Arc<dyn gb_vision::preprocessor::ImagePreprocessor>,
    pub gate: Arc<crate::preparation::PreparationGate>,
    pub ingress: Arc<crate::vision_ingress::HttpsIngress>,
}

impl Default for VisionHandle {
    fn default() -> Self {
        Self {
            slot: Default::default(),
            limits: Default::default(),
            pp: Arc::new(gb_vision::Preprocessing::qwen38()),
            gate: crate::preparation::PreparationGate::new(Default::default()).expect("valid preparation defaults"),
            ingress: crate::vision_ingress::HttpsIngress::new(Default::default()).expect("valid ingress defaults"),
        }
    }
}

/// New tokens a request may generate: its own `max_tokens` (or `max_completion_tokens`) if given, else `default_cap`
/// (0 = no cap), always within the `room` left in the context window.
pub fn length_budget(room: usize, requested: Option<usize>, default_cap: usize) -> usize {
    match requested {
        Some(m) => m.min(room),
        None if default_cap > 0 => default_cap.min(room),
        None => room,
    }
}

/// Budgeted JSON extraction. The lease follows the Value until preparation buffers are destroyed.
pub struct ApiJson(pub Arc<Value>, pub Arc<crate::budget::RequestLease>);

impl axum::extract::FromRequest<Arc<AppState>> for ApiJson {
    type Rejection = Response;
    async fn from_request(req: axum::extract::Request, state: &Arc<AppState>) -> Result<Self, Response> {
        use crate::budget::{too_large, validate_json, RequestLease};
        let limits = &state.request_budget.limits;
        let content_type = req.headers().get(axum::http::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("");
        let mime = content_type.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
        if mime != "application/json" && !(mime.starts_with("application/") && mime.ends_with("+json")) {
            return Err(err(ApiError::status(
                415,
                "invalid_request_error",
                "the request needs the header `Content-Type: application/json`",
            )));
        }
        let declared = req
            .headers()
            .get(axum::http::header::CONTENT_LENGTH)
            .map(|v| {
                v.to_str().ok().and_then(|v| v.parse::<usize>().ok()).ok_or_else(|| err(ApiError::bad("invalid Content-Length")))
            })
            .transpose()?;
        if declared.is_some_and(|n| n > limits.body_bytes) {
            return Err(err(too_large(format!("the request body is too large (the limit is {} MiB)", limits.body_bytes >> 20))));
        }
        let mut raw = state.request_budget.lease();
        raw.grow_to(declared.unwrap_or(0)).map_err(err)?;
        let mut bytes: Vec<u8> = Vec::new();
        let mut stream = req.into_body().into_data_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| err(ApiError::bad("the request body could not be read")))?;
            let next = bytes.len().checked_add(chunk.len()).filter(|n| *n <= limits.body_bytes).ok_or_else(|| {
                err(too_large(format!("the request body is too large (the limit is {} MiB)", limits.body_bytes >> 20)))
            })?;
            let capacity = next.max(declared.unwrap_or(0));
            raw.grow_to(capacity).map_err(err)?;
            if capacity > bytes.capacity() {
                bytes.reserve_exact(capacity - bytes.len());
            }
            bytes.extend_from_slice(&chunk);
        }
        if declared.is_some_and(|n| n != bytes.len()) {
            return Err(err(ApiError::bad("Content-Length does not match the body")));
        }
        let preparation = state.request_budget.prepare_admitted().await.map_err(err)?;
        let (nodes, depth) = (limits.json_nodes, limits.json_depth);
        let job = (bytes, RequestLease::new(raw, preparation));
        tokio::task::spawn_blocking(move || {
            let _whole_owner = &job;
            let value = crate::tokenizer::off_workers(|| {
                validate_json(&job.0, nodes, depth)?;
                serde_json::from_slice::<Value>(&job.0).map_err(|_| ApiError::bad("the request body is not valid JSON"))
            })?;
            let (bytes, lease) = job;
            drop(bytes);
            Ok::<_, ApiError>(Self(Arc::new(value), Arc::new(lease)))
        })
        .await
        .map_err(|_| err(ApiError::status(500, "server_error", "JSON preparation failed")))?
        .map_err(err)
    }
}

/// Any other route: the same shape as every other refusal.
async fn no_such_route(req: axum::extract::Request) -> Response {
    err(ApiError::status(404, "invalid_request_error", format!("no such route: {} {}", req.method(), req.uri().path())))
}

/// A known route with a method it does not have (axum adds the `Allow` header to a 405 itself).
async fn wrong_method(req: axum::extract::Request) -> Response {
    err(ApiError::status(405, "invalid_request_error", format!("{} is not allowed on {}", req.method(), req.uri().path())))
}

#[derive(Clone)]
pub struct RequestDeadline(pub Instant, pub FailureHook);
async fn deadline_layer(
    State(s): State<Arc<AppState>>,
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if req.method() != axum::http::Method::POST {
        return next.run(req).await;
    }
    let at = Instant::now() + s.handle.request_timeout;
    let seen = Arc::new(std::sync::atomic::AtomicU8::new(0));
    let metrics = s.handle.metrics.clone();
    let hook: FailureHook = Arc::new(move |error| {
        let (bit, counter) = match error.kind {
            FailKind::RequestTimeout => (1, &metrics.request_timeouts),
            FailKind::Backpressure => (2, &metrics.stream_failures),
            _ => return,
        };
        if seen.fetch_or(bit, Ordering::AcqRel) & bit == 0 {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    });
    req.extensions_mut().insert(RequestDeadline(at, hook.clone()));
    match tokio::time::timeout_at(at.into(), next.run(req)).await {
        Ok(r) => {
            // The handler's worker-owned deadline may win the same-instant race. Observe the
            // actual overall expiry here too; the shared hook guarantees exactly one count.
            if r.status() == StatusCode::GATEWAY_TIMEOUT && Instant::now() >= at {
                hook(&SeqError { kind: FailKind::RequestTimeout, message: "the end-to-end request deadline expired".into() });
            }
            r
        }
        Err(_) => {
            hook(&SeqError { kind: FailKind::RequestTimeout, message: "the end-to-end request deadline expired".into() });
            err(ApiError::status(504, "request_timeout", "the end-to-end request deadline expired"))
        }
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    // every encode of this server's tokenizer is charged against the processing reserve (`budget::TokenizerGate`)
    state.tok.charge_to(state.request_budget.tokenizer_gate().clone());
    Router::new()
        .route("/health/live", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/health", get(health))
        .route("/health/ready", get(health))
        .route("/v1/health", get(health))
        .route("/v1/models", get(models))
        .route("/metrics", get(metrics))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/chat/completions/render", post(render))
        .route("/v1/completions", post(completions))
        .route("/v1/decisions", post(decisions))
        .route("/tokenize", post(tokenize))
        .fallback(no_such_route)
        .method_not_allowed_fallback(wrong_method)
        // images arrive as base64 data URIs: the default 2 MiB body limit would refuse a single photo
        .layer(axum::extract::DefaultBodyLimit::disable())
        .layer(axum::middleware::from_fn_with_state(state.clone(), deadline_layer))
        .layer(axum::middleware::from_fn_with_state(state.clone(), crate::observability::middleware))
        .with_state(state)
}

fn err(e: ApiError) -> Response {
    let mut r = (StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_REQUEST), Json(e.body())).into_response();
    if let Some(secs) = e.retry_after {
        r.headers_mut().insert(axum::http::header::RETRY_AFTER, axum::http::HeaderValue::from(secs));
    }
    r
}

/// Readiness: 200 only while the engine is up and healthy. 503 (with the reason in `status`) while it loads, while it drains for a
/// shutdown, after consecutive step errors (`degraded`), and once the engine has stopped (`failed`).
/// Native prompt-only score jobs, one question at a time; no partial responses.
struct PreparedDecisions {
    questions: Vec<crate::decisions::PreparedDecision>,
    _credit: Arc<tokio::sync::OwnedSemaphorePermit>,
}
async fn decisions(
    State(s): State<Arc<AppState>>,
    Extension(RequestDeadline(request_deadline, _)): Extension<RequestDeadline>,
    request: ApiJson,
) -> Response {
    if let Err(e) = crate::field_policy::validate_model(&request.0, &s.model_name, &s.model_aliases) {
        return err(e);
    }
    match s.handle.metrics.state() {
        crate::runtime::State::Loading => return err(refusal_error(Refusal::NotReady, s.max_model_len)),
        crate::runtime::State::Draining => return err(refusal_error(Refusal::ShuttingDown, s.max_model_len)),
        crate::runtime::State::Failed => {
            return err(refusal_error(Refusal::EngineDown("the engine has stopped".into()), s.max_model_len))
        }
        _ => {}
    }
    if !s.handle.metrics.ready_complete() {
        return err(refusal_error(Refusal::NotReady, s.max_model_len));
    }
    if !s.handle.score_supported() {
        return err(refusal_error(Refusal::Rejected(gb_engine::score::unsupported()), s.max_model_len));
    }
    let _decision_owner = match s.request_budget.decision_admitted() {
        Ok(permit) => Arc::new(permit),
        Err(e) => return err(e),
    };
    let deadline: tokio::time::Instant = request_deadline.min(Instant::now() + s.handle.decision_timeout()).into();
    let limit = s.handle.metrics.effective_limit(s.max_model_len);
    let work = crate::preparation::RequestWork::new(deadline.saturating_duration_since(tokio::time::Instant::now()));
    let _cancel = crate::preparation::CancelOnDrop(work.clone());
    let state = s.clone();
    let inputs = (request.0.clone(), request.1.clone(), _decision_owner.clone());
    let prepared = match tokio::task::spawn_blocking(move || {
        let _owner = &inputs;
        work.check()?;
        let questions = crate::tokenizer::off_workers(|| {
            crate::decisions::prepare_capped(
                &state.tok,
                &*state.format,
                (*inputs.0).clone(),
                limit,
                state.request_budget.limits.rendered_bytes,
            )
        })?;
        work.check()?;
        Ok::<_, ApiError>(PreparedDecisions { questions, _credit: inputs.2.clone() })
    })
    .await
    {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => return err(e),
        Err(_) => return err(ApiError::status(500, "server_error", "decision preparation failed")),
    };
    request.1.finish_preparation(); // Worker really joined; retain independent raw/body + decision credits.
    let mut body = (*request.0).clone();
    body["model"] = json!(s.model_name);
    let mut rows = Vec::with_capacity(prepared.questions.len());
    for q in &prepared.questions {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let result = tokio::time::timeout_at(deadline, async {
            let (guard, rx) = s
                .handle
                .score_with_timeout(
                    gb_engine::score::ScoreRequest { prompt: q.prompt.clone(), cache: true, all_rows: false },
                    remaining,
                )
                .await
                .map_err(|e| refusal_error(e, limit))?;
            let row = rx
                .await
                .map_err(|_| ApiError::status(503, "server_error", "the score engine stopped"))?
                .map_err(|e| seq_error(&e))?;
            drop(guard);
            Ok::<_, ApiError>(crate::decisions::ScoredQuestion { logits_row: row.logits_row, cached_tokens: row.cached_tokens })
        })
        .await;
        match result {
            Ok(Ok(r)) => rows.push(r),
            Ok(Err(e)) => return err(e),
            Err(_) => {
                return err(ApiError {
                    code: Some("request_timeout"),
                    ..ApiError::status(504, "request_timeout", "decision request deadline exceeded")
                })
            }
        }
    }
    match crate::decisions::build_response(&body, &prepared.questions, &rows) {
        Ok(v) => Json(v).into_response(),
        Err(e) => err(e),
    }
}

async fn health(State(s): State<Arc<AppState>>) -> Response {
    let state = s.handle.metrics.state();
    // rank 0 of a tensor-parallel group reports its followers too (state, last ack, lag, ping); a single process adds nothing
    let group = gb_model::tp_status::snapshot().map(|g| crate::tp_metrics::health(&g));
    let body = |status: &str| {
        let mut v = json!({"status": status});
        if let Some(g) = &group {
            v["tensor_parallel"] = g.clone();
        }
        v
    };
    let lost = group.as_ref().is_some_and(|g| g["state"] == "lost");
    let (code, word) = health_verdict(
        s.handle.metrics.wedged(),
        lost,
        s.handle.metrics.ready(),
        s.handle.metrics.ready_complete(),
        state.name(),
    );
    (code, Json(body(word))).into_response()
}

/// The status code and `status` word of `/health` and `/health/ready`. A lost tensor-parallel group is not ready even while the engine,
/// idle, has not noticed (it ends the process only at the next call or at the 20 s exit backstop): found on the real pair, 2026-10-11,
/// where an idle leader answered 200 for 20 s after its follower was killed.
fn health_verdict(
    wedged: bool,
    group_lost: bool,
    ready: bool,
    ready_complete: bool,
    state: &'static str,
) -> (StatusCode, &'static str) {
    if wedged {
        (StatusCode::SERVICE_UNAVAILABLE, "wedged")
    } else if !ready || !ready_complete {
        (StatusCode::SERVICE_UNAVAILABLE, if ready_complete { state } else { "loading" })
    } else if group_lost {
        (StatusCode::SERVICE_UNAVAILABLE, "group_lost")
    } else {
        (StatusCode::OK, state)
    }
}

/// The KV pool the plan allocated, in tokens (the boot fact `kv_tokens`), once the engine has planned it.
fn kv_pool_tokens() -> Option<u64> {
    gb_engine::boot::facts().into_iter().find(|(n, _)| *n == "kv_tokens").and_then(|(_, v)| v.parse().ok())
}

/// The longest prefix the prefix cache can ever serve, in tokens (the boot fact `prefix_cache_reach_tokens`); `None` for a family that does
/// not report it or a server running without a cache.
fn prefix_cache_reach_tokens() -> Option<u64> {
    gb_engine::boot::facts().into_iter().find(|(n, _)| *n == "prefix_cache_reach_tokens").and_then(|(_, v)| v.parse().ok())
}

async fn models(State(s): State<Arc<AppState>>) -> Json<Value> {
    // `max_model_len` is what a client decides whether a prompt fits on: the smaller of the configured window and what the memory plan can hold
    // for one request. The window as configured stays visible beside it, and so does the pool the limit came from.
    Json(json!({"object": "list", "data": [{
        "id": s.model_name, "object": "model", "created": api::now(), "owned_by": "rexl3",
        "max_model_len": s.handle.metrics.effective_limit(s.max_model_len),
        "configured_max_model_len": s.max_model_len,
        "kv_pool_tokens": kv_pool_tokens(),
        // what the prefix cache can serve (a prompt longer than this is never a hit); null when the family does not report it
        "prefix_cache_reach_tokens": prefix_cache_reach_tokens(),
        "pack_validated": gb_engine::boot::facts().into_iter().find(|(n, _)| *n == "pack_validated").map(|(_, v)| v),
        // a pack served with --allow-unvalidated-pack: the boot's warning line (`gb_model::serve::UNVALIDATED_PACK_WARNING`); null otherwise
        "pack_warning": gb_engine::boot::facts().into_iter().find(|(n, _)| *n == "pack_warning").map(|(_, v)| v),
        // what reasoning/thinking controls the model offers (docs/REASONING_CAPABILITIES.md)
        "reasoning": s.reasoning_caps.exposure(s.default_thinking),
    }]}))
}

/// The `system_fingerprint` every response carries (see [`crate::build_info::BuildInfo::fingerprint`]).
fn fp() -> &'static str {
    static FP: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    FP.get_or_init(|| crate::build_info::BuildInfo::get().fingerprint())
}

/// Prometheus label value: backslash, quote and newline escaped.
fn label(v: &str) -> String {
    v.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

async fn metrics(State(s): State<Arc<AppState>>) -> String {
    let m = &s.handle.metrics;
    let g = |c: &std::sync::atomic::AtomicU64| c.load(Ordering::Relaxed);
    let mut out = String::new();
    let mut put = |name: &str, kind: &str, help: &str, samples: &[(String, String)]| {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n"));
        for (labels, v) in samples {
            out.push_str(&format!("{name}{labels} {v}\n"));
        }
    };
    let one = |v: u64| vec![(String::new(), v.to_string())];
    put(
        "rexl3_decisions_total",
        "counter",
        "Decision question score jobs by terminal/admission status (not HTTP request count).",
        &[
            ("{status=\"admitted\"}".into(), g(&m.decisions_admitted).to_string()),
            ("{status=\"completed\"}".into(), g(&m.decisions_completed).to_string()),
            ("{status=\"cancelled\"}".into(), g(&m.decisions_cancelled).to_string()),
        ],
    );
    put(
        "rexl3_decision_prompt_tokens_total",
        "counter",
        "Prompt tokens admitted to native score jobs.",
        &one(g(&m.decision_prompt_tokens)),
    );
    put(
        "rexl3_decisions_cached_tokens_total",
        "counter",
        "Prefix tokens reused by completed native score jobs.",
        &one(g(&m.decisions_cached_tokens)),
    );
    let b = crate::build_info::BuildInfo::get();
    let info: Vec<String> = b.lines().iter().map(|(k, v)| format!("{k}=\"{}\"", label(v))).collect();
    put("rexl3_build_info", "gauge", "Build identity (labels); always 1.", &[(format!("{{{}}}", info.join(",")), "1".into())]);
    let facts: Vec<String> = gb_engine::boot::facts().iter().map(|(k, v)| format!("{k}=\"{}\"", label(v))).collect();
    put(
        "rexl3_engine_info",
        "gauge",
        "Pack fingerprints, validation and KV plan (labels); always 1.",
        &[(format!("{{{}}}", facts.join(",")), "1".into())],
    );
    let rl: Vec<String> =
        s.reasoning_caps.metric_labels(s.default_thinking).iter().map(|(k, v)| format!("{k}=\"{}\"", label(v))).collect();
    put(
        "rexl3_reasoning_info",
        "gauge",
        "The model's reasoning capabilities as checked at load (labels); always 1.",
        &[(format!("{{{}}}", rl.join(",")), "1".into())],
    );
    let phases = gb_engine::boot::phases();
    put(
        "rexl3_boot_phase_seconds",
        "gauge",
        "Seconds from process start to each boot phase.",
        &phases.iter().map(|p| (format!("{{phase=\"{}\"}}", p.name), format!("{:.3}", p.secs))).collect::<Vec<_>>(),
    );
    put(
        "rexl3_boot_phase_gib",
        "gauge",
        "GiB made resident by each boot phase that has a size.",
        &phases
            .iter()
            .filter_map(|p| p.gib.map(|gb| (format!("{{phase=\"{}\"}}", p.name), format!("{gb:.3}"))))
            .collect::<Vec<_>>(),
    );
    let rows = gb_model::loadstats::snapshot();
    let cell = |r: &gb_model::loadstats::Row| format!("{{component=\"{}\",stage=\"{}\"}}", r.component.name(), r.stage.name());
    put(
        "rexl3_load_stage_seconds",
        "gauge",
        "Seconds the weight loader spent in each stage (read, convert, malloc, memcpy, memset, wait, pin) per component (scan, target, draft, vision, engine). Summed over loader threads: equals wall time while one thread loads, more when several do.",
        &rows.iter().map(|r| (cell(r), format!("{:.4}", r.seconds))).collect::<Vec<_>>(),
    );
    put(
        "rexl3_load_stage_bytes",
        "gauge",
        "Bytes each weight-loader stage moved or converted.",
        &rows.iter().map(|r| (cell(r), r.bytes.to_string())).collect::<Vec<_>>(),
    );
    put(
        "rexl3_load_stage_calls",
        "gauge",
        "Calls each weight-loader stage made.",
        &rows.iter().map(|r| (cell(r), r.calls.to_string())).collect::<Vec<_>>(),
    );
    put(
        "rexl3_engine_heartbeat_age_seconds",
        "gauge",
        "Age of last safe engine progress, with phase-specific readiness budgets.",
        &[(String::new(), m.heartbeat_age().as_secs_f64().to_string())],
    );
    put("rexl3_request_timeouts_total", "counter", "End-to-end request deadlines expired.", &one(g(&m.request_timeouts)));
    // only once it has happened (an engine that states no prefill bound never refuses on it): the default text is unchanged
    if g(&m.deadline_refusals) > 0 {
        put(
            "rexl3_request_deadline_refusals_total",
            "counter",
            "Requests refused at once because even a lower bound on their prefill exceeded their deadline.",
            &one(g(&m.deadline_refusals)),
        );
    }
    put(
        "rexl3_stream_failures_total",
        "counter",
        "Terminal bounded-stream producer failures (not HTTP header success).",
        &one(g(&m.stream_failures)),
    );
    put("rexl3_ready", "gauge", "1 while the engine is up and healthy (what /health reports).", &one(m.ready() as u64));
    put(
        "rexl3_engine_state",
        "gauge",
        "Engine state, one series per state, 1 for the current one (loading, ok, draining, degraded, failed).",
        &["loading", "ok", "draining", "degraded", "failed"]
            .iter()
            .map(|n| (format!("{{state=\"{n}\"}}"), if m.state().name() == *n { "1" } else { "0" }.to_string()))
            .collect::<Vec<_>>(),
    );
    put(
        "rexl3_uptime_seconds",
        "gauge",
        "Seconds since the process started.",
        &[(String::new(), format!("{:.1}", gb_engine::boot::elapsed()))],
    );
    put("rexl3_requests_total", "counter", "Requests admitted.", &one(g(&m.requests)));
    put(
        "rexl3_requests_rejected_total",
        "counter",
        "Requests that never ran, counted once each: queue full, can never fit, malformed, server shutting down, queue timeout.",
        &one(g(&m.rejected)),
    );
    put(
        "rexl3_requests_queue_full_total",
        "counter",
        "Requests refused with 429 because the queue was full.",
        &one(g(&m.queue_full)),
    );
    put(
        "rexl3_queue_timeouts_total",
        "counter",
        "Requests answered 503 after waiting longer than --queue-timeout.",
        &one(g(&m.queue_timeouts)),
    );
    put(
        "rexl3_requests_cancelled_total",
        "counter",
        "Engine cancellations before normal engine completion, including client departures and server stop-string cuts; not server failures.",
        &one(g(&m.cancelled)),
    );
    put("rexl3_queue_depth", "gauge", "Requests waiting for a free slot.", &one(g(&m.queued)));
    put(
        "rexl3_prefilling_sequences",
        "gauge",
        "Admitted sequences whose prompt is still being prefilled.",
        &one(g(&m.prefilling)),
    );
    put(
        "rexl3_failed_sequences_total",
        "counter",
        "Sequences the engine failed (a bad row, a prefill error, a failed step).",
        &one(g(&m.failed_sequences)),
    );
    put(
        "rexl3_prompt_tokens_total",
        "counter",
        "Full prompt tokens of admitted requests, including cached prefixes; not actual prefill work.",
        &one(g(&m.prompt_tokens)),
    );
    put(
        "rexl3_completion_tokens_total",
        "counter",
        "Tokens produced in real engine token batches; not SSE events or delivered client tokens.",
        &one(g(&m.completion_tokens)),
    );
    put("rexl3_spec_drafted_total", "counter", "Draft tokens proposed, accumulated when requests finish.", &one(g(&m.drafted)));
    put("rexl3_spec_accepted_total", "counter", "Draft tokens accepted, accumulated when requests finish.", &one(g(&m.accepted)));
    put("rexl3_spec_rounds_total", "counter", "Verify rounds of finished requests.", &one(g(&m.rounds)));
    put("rexl3_active_sequences", "gauge", "Sequences decoding.", &one(g(&m.active)));
    put("rexl3_step_errors_total", "counter", "Engine rounds that returned an error.", &one(g(&m.step_errors)));
    let limit = m.effective_limit(s.max_model_len);
    put(
        "rexl3_max_model_len",
        "gauge",
        "Longest prompt plus generation one request may hold: the smaller of the configured window and the KV pool the plan allocated.",
        &one(limit as u64),
    );
    put(
        "rexl3_configured_max_model_len",
        "gauge",
        "The context window as configured (--max-model-len).",
        &one(s.max_model_len as u64),
    );
    // KV usage/capacity are emitted together from one declared engine snapshot by observability::render.
    // Do not independently render boot facts here: engine boot racing this scrape could duplicate the family.
    // the vLLM-style label r0b0bench's NIAH lane reads to size its haystacks: what one prompt can really be, so it is the effective limit
    put(
        "rexl3_cache_config_info",
        "gauge",
        "The per-request limit as a kv_cache_size_tokens label (read by r0b0bench) and the configured window; always 1.",
        &[(format!("{{kv_cache_size_tokens=\"{limit}\",configured_max_model_len=\"{}\"}}", s.max_model_len), "1".into())],
    );
    if let Some(v) = &s.vision {
        let gate = &v.gate;
        put(
            "rexl3_image_retained_bytes",
            "gauge",
            "Image input byte credits including queued work.",
            &one(gate.retained_bytes() as u64),
        );
        put(
            "rexl3_image_preparation_jobs",
            "gauge",
            "Admitted image preparation jobs including bounded waiters.",
            &one(gate.jobs() as u64),
        );
        put("rexl3_image_preparation_bytes", "gauge", "CPU preparation byte credits held by workers.", &one(gate.active_bytes()));
        put("rexl3_image_gate_wait_count", "counter", "Image preparation admission attempts.", &one(g(&gate.wait_count)));
        put(
            "rexl3_image_gate_wait_max_seconds",
            "gauge",
            "Maximum image preparation gate wait seconds.",
            &[(String::new(), format!("{:.6}", g(&gate.wait_max_micros) as f64 / 1e6))],
        );
        put("rexl3_image_gate_rejected_total", "counter", "Image preparation admission refusals.", &one(g(&gate.rejected)));
        put(
            "rexl3_image_preparation_cancelled_total",
            "counter",
            "Blocking image jobs cooperatively stopped after cancellation.",
            &one(g(&gate.cancelled)),
        );
        put("rexl3_images_processed_total", "counter", "Successfully encoded images.", &one(g(&gate.processed)));
        put(
            "rexl3_image_preparation_timeout_total",
            "counter",
            "Image preparation hard deadline expirations.",
            &one(g(&gate.timed_out)),
        );
        put(
            "rexl3_image_fetch_total",
            "counter",
            "Remote image fetch results.",
            &[
                ("{result=\"ok\"}".into(), g(&v.ingress.ok).to_string()),
                ("{result=\"denied\"}".into(), g(&v.ingress.denied).to_string()),
                ("{result=\"timeout\"}".into(), g(&v.ingress.timeout).to_string()),
                ("{result=\"oversize\"}".into(), g(&v.ingress.oversize).to_string()),
                ("{result=\"bad_type\"}".into(), g(&v.ingress.bad_type).to_string()),
            ],
        );
    }
    if let Ok(spec) = m.spec.lock() {
        out.push_str(&crate::spec_metrics::render(&spec));
    }
    if let Ok(px) = m.prefix.lock() {
        out.push_str(&crate::prefix_metrics::render(&px));
    }
    if let Some(g) = m.graphs.lock().ok().and_then(|g| *g) {
        out.push_str(&crate::graph_metrics::render(&g));
    }
    if let Some(g) = gb_model::tp_status::snapshot() {
        out.push_str(&crate::tp_metrics::render(&g, gb_model::tp_status::groups_lost()));
    }
    m.observability.render(&mut out);
    out
}

/// What the generation task sends to the response writer.
enum Out {
    Text {
        reasoning: String,
        content: String,
        ids: Vec<u32>,
        calls: Vec<crate::tools::ToolCall>,
    },
    /// The end of the reply. `error` is set when it did not finish (the engine failed it, the queue timed out, the server shut
    /// down): the reply is cut short and must be reported as an error, never as a normal stop.
    Done {
        finish: &'static str,
        usage: Value,
        error: Option<SeqError>,
    },
}

impl Message for Out {
    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + match self {
                Self::Text { reasoning, content, ids, calls } => reasoning
                    .capacity()
                    .saturating_add(content.capacity())
                    .saturating_add(ids.capacity().saturating_mul(4))
                    .saturating_add(
                        calls.capacity().saturating_mul(std::mem::size_of::<crate::tools::ToolCall>()).saturating_add(
                            calls.iter().map(|c| c.name.capacity().saturating_add(c.arguments.capacity())).sum::<usize>(),
                        ),
                    ),
                Self::Done { usage, error, .. } => {
                    usage.to_string().len().saturating_add(error.as_ref().map_or(0, |e| e.message.capacity()))
                }
            }
    }
    fn failure(e: SeqError) -> Self {
        Self::Done { finish: "stop", usage: Value::Null, error: Some(e) }
    }
}
struct WireEvent(String, Option<bool>);
impl Message for WireEvent {
    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.0.capacity()
    }
    fn failure(e: SeqError) -> Self {
        Self(error_event(&e).to_string(), Some(true))
    }
}

/// The API error a request that did not finish is answered with (non-streaming), or reports in its stream's last event.
fn seq_error(e: &SeqError) -> ApiError {
    match e.kind {
        FailKind::RequestTimeout => {
            ApiError { code: Some("request_timeout"), ..ApiError::status(504, "request_timeout", e.message.clone()) }
        }
        FailKind::Engine => ApiError::status(500, "server_error", e.message.clone()),
        FailKind::QueueTimeout => ApiError::status(503, "server_overloaded", e.message.clone()).retry_after(5),
        FailKind::Shutdown => ApiError::status(503, "server_shutting_down", e.message.clone()),
        FailKind::EngineDown => ApiError::status(503, "service_unavailable", e.message.clone()),
        FailKind::Backpressure => ApiError::status(503, "server_overloaded", e.message.clone()),
    }
}

/// A stream's closing error event.
fn error_event(e: &SeqError) -> Value {
    seq_error(e).body()
}

/// Holds back text that might begin a stop string; cuts at the first full match.
struct StopMatcher {
    stops: Vec<String>,
    held: String,
}

impl StopMatcher {
    fn new(stops: Vec<String>) -> Self {
        Self { stops, held: String::new() }
    }
    /// Returns (text safe to emit, whether a stop string matched).
    fn push(&mut self, text: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (text.to_string(), false);
        }
        self.held.push_str(text);
        if let Some(cut) = self.stops.iter().filter_map(|s| self.held.find(s.as_str())).min() {
            let out = self.held[..cut].to_string();
            self.held.clear();
            return (out, true);
        }
        let keep = self.stops.iter().map(|s| s.len().saturating_sub(1)).max().unwrap_or(0).min(self.held.len());
        let mut cut = self.held.len() - keep;
        while !self.held.is_char_boundary(cut) {
            cut += 1;
        }
        let out = self.held[..cut].to_string();
        self.held = self.held[cut..].to_string();
        (out, false)
    }
    fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

fn gib(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
    } else {
        format!("{:.1} MiB", bytes as f64 / (1u64 << 20) as f64)
    }
}

/// The text of a `context_length_exceeded` refusal. `limit` is what one request may hold (the smaller of the window and the KV pool the plan
/// allocated); when it is below the configured `window`, the message says where the limit comes from, so a client that read the window is not
/// left wondering why a prompt under it is refused.
fn context_message(limit: usize, window: usize, detail: &str) -> String {
    let mut m = format!("This model's maximum context length is {limit} tokens. However, {detail}.");
    if limit < window {
        m.push_str(&format!(
            " The KV cache this server planned at start-up holds {limit} tokens (the configured context window is {window}); \
             a longer prompt needs a server started with more KV memory"
        ));
    }
    m
}

fn submit_error(e: SubmitError, max_model_len: usize) -> ApiError {
    match e {
        SubmitError::Invalid(m) => ApiError::bad(m),
        SubmitError::Sampling(e) => {
            ApiError::status(500, "server_error", format!("the model produced an unusable logit row: {e}"))
        }
        SubmitError::Admit(AdmitError::ContextTooLong { tokens, max }) => {
            ApiError::context_length(context_message(max as usize, max_model_len, &format!("you requested {tokens} tokens in total")))
        }
        SubmitError::Backend(m) => ApiError::status(500, "server_error", format!("the engine failed this request: {m}")),
        SubmitError::Admit(AdmitError::TooLarge) => ApiError::bad(format!("request is too large (model length {max_model_len})")),
        SubmitError::Admit(AdmitError::NeverFits { needed, capacity }) => ApiError::bad(format!(
            "this request needs {} of KV cache and sequence state, but the server can give a single sequence at most {}: shorten the \
             prompt or lower max_tokens (the context window is {max_model_len} tokens)",
            gib(needed),
            gib(capacity)
        )),
        SubmitError::Admit(a) => ApiError::status(503, "server_overloaded", format!("no capacity for this request: {a:?}")).retry_after(5),
    }
}

/// `Retry-After` for a request refused because the queue is full (seconds).
const RETRY_AFTER_QUEUE_FULL: u32 = 2;

/// What `EngineHandle::submit` refusing a request means to the client.
fn refusal_error(r: Refusal, max_model_len: usize) -> ApiError {
    match r {
        Refusal::QueueFull { depth } => {
            ApiError::status(429, "server_overloaded", format!("the request queue is full ({depth} waiting); retry shortly"))
                .retry_after(RETRY_AFTER_QUEUE_FULL)
        }
        Refusal::NotReady => ApiError::status(503, "service_unavailable", "the engine is still loading").retry_after(5),
        Refusal::ShuttingDown => ApiError::status(503, "server_shutting_down", "the server is shutting down"),
        Refusal::EngineDown(m) => ApiError::status(503, "service_unavailable", format!("the engine has stopped: {m}")),
        Refusal::Rejected(e) => submit_error(e, max_model_len),
        Refusal::RequestTimeout => ApiError::status(504, "request_timeout", "the request deadline expired before admission"),
        Refusal::DeadlineTooShort { floor, left } => ApiError::bad(format!(
                "this prompt needs at least {:.0} s of prefill on this server and its deadline leaves {:.0} s (--request-timeout): shorten the \
                 prompt, or ask the operator for a longer --request-timeout",
                floor.as_secs_f64(),
                left.as_secs_f64()
        ))
        .with_code("request_deadline_too_short"),
    }
}

/// A request that is queued or running: the receiver of its reply, its prompt length, and the guard that ends the request when the
/// client goes away (dropped with whoever holds the reply: the handler, or the response body of a stream).
struct Started {
    rx: Receiver<Out>,
    plen: usize,
    guard: CancelGuard,
    failure_hook: FailureHook,
}

/// Validate the length budget, queue the request (it waits in the engine's FIFO queue for a slot; a full queue, or a request that can
/// never fit, is refused here with its status code), and start the task that turns engine tokens into text.
#[allow(clippy::too_many_arguments)]
async fn start(
    s: &Arc<AppState>,
    (arrival, RequestDeadline(deadline, failure_hook)): (Arrival, RequestDeadline),
    prompt: Vec<u32>,
    p: &Params,
    starts_in_reasoning: bool,
    images: Vec<gb_engine::ImageSpan>,
    tools: Option<crate::tools::ToolSchemas>,
    reasoning_budget: Option<gb_engine::ReasoningBudget>,
) -> Result<Started, ApiError> {
    let plen = prompt.len();
    if plen == 0 {
        return Err(ApiError::bad("the prompt is empty"));
    }
    // what one request may hold is the smaller of the window and the KV pool the plan allocated: a prompt the pool cannot hold is refused here,
    // at once, instead of being admitted and prefilled until the pool runs out; and the generation budget is clipped to the room the pool leaves
    let limit = s.handle.metrics.effective_limit(s.max_model_len);
    if plen >= limit {
        return Err(ApiError::context_length(context_message(
            limit,
            s.max_model_len,
            &format!("your request has {plen} prompt tokens"),
        )));
    }
    let room = limit - plen;
    let max_new = length_budget(room, p.max_tokens, s.default_max_tokens);
    let mut sampling = p.sampling;
    if p.seed.is_none() {
        sampling.seed = gb_sampling::seed_for(&prompt, 0); // same conversation, same reply
    }
    let (tx, mut rx) = channel_with_failure::<SeqMsg>(Some(failure_hook.clone()));
    let req = GenRequest {
        prompt,
        sampling,
        max_new,
        stop: s.tok.eos.clone(),
        draft: p.draft,
        images,
        cache: p.prefix_cache,
        reasoning_budget,
    };
    let submit_at = std::time::Instant::now();
    s.handle.metrics.observability.preprocessing.record(submit_at.duration_since(arrival.0));
    let submitted = s.handle.submit_until(req, tx, deadline).await;
    // the engine has answered: this request no longer holds an idle engine's gather window open
    arrival.1.settle(&s.handle.metrics.observability);
    let ticket = submitted.map_err(|r| refusal_error(r, s.max_model_len))?;
    let guard = CancelGuard::new(s.handle.clone(), ticket);
    let (otx, orx) = channel_with_failure::<Out>(Some(failure_hook.clone()));
    let tok = s.tok.clone();
    let format = s.format.clone();
    let reasoning_caps = s.reasoning_caps.clone();
    let handle = s.handle.clone();
    let stops = p.stop.clone();
    let skip = p.skip_special;
    let want_ids = p.return_token_ids;
    let stall = s.handle.write_stall;
    tokio::spawn(async move {
        let mut dec = StreamDecoder::new(skip);
        let mut rs = ReasoningSplitter::with_close(starts_in_reasoning, reasoning_caps.close());
        let mut thinking = ReasoningCount::new(starts_in_reasoning, reasoning_caps.close_id);
        let mut stopper = StopMatcher::new(stops);
        let mut tool_split = tools.map(|t| format.tool_splitter(t));
        let mut n_calls = 0usize;
        let mut n_tokens = 0usize;
        let mut decoded_bytes = 0usize;
        let mut last_batch: Option<std::time::Instant> = None;
        let mut stopped_by_string = false;
        let mut info: Option<DoneInfo> = None;
        loop {
            // wake when the client is gone as well as when the engine speaks: while the request is queued or prefilling nothing
            // else would notice, and the slot would be held until its first token
            let msg = tokio::select! {
                m = rx.recv() => m,
                _ = tokio::time::sleep_until(deadline.into()) => {
                    handle.cancel(ticket);
                    otx.fail(SeqError { kind: FailKind::RequestTimeout, message: "the request deadline expired".into() });
                    return;
                },
                _ = otx.closed() => {
                    handle.cancel(ticket);
                    return;
                }
            };
            let Some(msg) = msg else { break };
            match msg {
                SeqMsg::Tokens(ids) => {
                    let now = std::time::Instant::now();
                    let o = &handle.metrics.observability;
                    if let Some(last) = last_batch {
                        o.batch_gap.record(now.duration_since(last));
                    } else {
                        o.ttft.record(now.duration_since(arrival.0));
                        o.submit_ttft.record(now.duration_since(submit_at));
                    }
                    last_batch = Some(now);
                    n_tokens += ids.len();
                    thinking.push(&ids);
                    let piece = dec.push(&tok, &ids);
                    decoded_bytes = decoded_bytes.saturating_add(piece.len());
                    if decoded_bytes > 8 << 20 {
                        handle.cancel(ticket);
                        otx.fail(SeqError {
                            kind: FailKind::Backpressure,
                            message: "the response text budget was exceeded".into(),
                        });
                        return;
                    }
                    let split = rs.push(&piece);
                    let (visible, calls) = match tool_split.as_mut() {
                        Some(t) => t.push(&split.content),
                        None => (split.content, Vec::new()),
                    };
                    n_calls += calls.len();
                    let (content, hit) = stopper.push(&visible);
                    let ids = if want_ids { ids } else { Vec::new() };
                    if (!split.reasoning.is_empty() || !content.is_empty() || !ids.is_empty() || !calls.is_empty())
                        && otx
                            .send_wait(Out::Text { reasoning: split.reasoning, content, ids, calls }, deadline, stall)
                            .await
                            .is_err()
                    {
                        handle.cancel(ticket); // the client left
                        return;
                    }
                    let dsml_done = tool_split.as_ref().is_some_and(|t| t.is_closed()) && n_calls > 0;
                    if hit || dsml_done {
                        stopped_by_string = true;
                        handle.cancel(ticket);
                        // The engine's Done arrives when it finishes the step it is on (cancel flushes an output). A fixed,
                        // short drain window loses that race when the step is long, and the fallback DoneInfo below then reports
                        // zero cached_tokens and zero speculation counters for a request that restored and generated normally.
                        // Wait for the Done for the rest of the request instead; the channel closing is the only early exit.
                        let remain = deadline.saturating_duration_since(std::time::Instant::now());
                        while let Ok(msg) = tokio::time::timeout(remain, rx.recv()).await {
                            if let Some(SeqMsg::Done(d)) = msg {
                                info = Some(d);
                                break;
                            } else if msg.is_none() {
                                break;
                            }
                        }
                        break;
                    }
                }
                SeqMsg::Done(d) => {
                    info = Some(d);
                    break;
                }
            }
        }
        if !stopped_by_string {
            let tail = dec.finish(&tok);
            let mut split = rs.push(&tail);
            let fin = rs.finish();
            split.reasoning.push_str(&fin.reasoning);
            split.content.push_str(&fin.content);
            let (mut visible, mut calls) = match tool_split.as_mut() {
                Some(t) => t.push(&split.content),
                None => (split.content, Vec::new()),
            };
            if let Some(t) = tool_split.as_mut() {
                let (v, c) = t.finish();
                visible.push_str(&v);
                calls.extend(c);
            }
            n_calls += calls.len();
            let (mut content, hit) = stopper.push(&visible);
            if hit {
                stopped_by_string = true;
            } else {
                content.push_str(&stopper.finish());
            }
            if (!split.reasoning.is_empty() || !content.is_empty() || !calls.is_empty())
                && otx
                    .send_wait(Out::Text { reasoning: split.reasoning, content, ids: Vec::new(), calls }, deadline, stall)
                    .await
                    .is_err()
            {
                handle.cancel(ticket);
                return;
            }
        }
        let (finish, error, d) = match info {
            Some(d) if stopped_by_string => {
                (api::finish_with_calls("stop", n_calls), None, DoneInfo { finish: None, error: None, ..d })
            }
            Some(d) => {
                let f = api::finish_str(d.finish, stopped_by_string);
                // a reply that ended after calling tools finishes as "tool_calls" (a length cut-off stays "length")
                (api::finish_with_calls(f, n_calls), d.error.clone(), d)
            }
            // we ended the request ourselves on a stop string or closed DSML tool call block: a normal stop
            None if stopped_by_string => (
                api::finish_with_calls("stop", n_calls),
                None,
                DoneInfo { finish: None, error: None, rounds: 0, drafted: 0, accepted: 0, cached_tokens: 0 },
            ),
            // the channel closed without an end: the engine is gone. This used to be reported as a normal stop (the zombie reply)
            None => (
                "stop",
                Some(SeqError { kind: FailKind::EngineDown, message: "the engine stopped before the reply was complete".into() }),
                DoneInfo { finish: None, error: None, rounds: 0, drafted: 0, accepted: 0, cached_tokens: 0 },
            ),
        };
        let usage = api::usage_json(plen, n_tokens, d.drafted, d.accepted, d.rounds, d.cached_tokens, thinking.tokens());
        if otx.send_wait(Out::Done { finish, usage, error }, deadline, stall).await.is_err() {
            handle.cancel(ticket);
        } else {
            handle.metrics.observability.e2e.record(arrival.0.elapsed());
        }
    });
    Ok(Started { rx: orx, plen, guard, failure_hook })
}

/// The streaming response: headers go out at once, the body is the events, and a comment (`: keep-alive`) every `keepalive` while
/// there is nothing else to send. The body owns `guard`: when the client goes away the body is dropped, which ends the request, wherever
/// it is (queued, prefilling or decoding).
fn sse_response(
    events: Receiver<WireEvent>,
    guard: CancelGuard,
    keepalive: Duration,
    deadline: Instant,
    metrics: Arc<crate::runtime::Metrics>,
) -> Response {
    let outcome = SseOutcome::new(metrics);
    let stream = futures_util::stream::unfold(
        (events, guard, false, outcome),
        move |(mut events, guard, ended, mut outcome)| async move {
            if ended {
                return None;
            }
            let (event, ended) = match tokio::time::timeout_at(deadline.into(), events.recv()).await {
                Ok(Some(e)) => (e, false),
                Ok(None) if !outcome.finished() => (
                    WireEvent::failure(SeqError {
                        kind: FailKind::EngineDown,
                        message: "the response producer ended before a terminal stream event".into(),
                    }),
                    true,
                ),
                Ok(None) => return None,
                Err(_) => (
                    WireEvent::failure(SeqError {
                        kind: FailKind::RequestTimeout,
                        message: "the request deadline expired".into(),
                    }),
                    true,
                ),
            };
            if let Some(error) = event.1 {
                outcome.terminal(error);
            }
            Some((Ok::<Event, Infallible>(Event::default().data(event.0)), (events, guard, ended, outcome)))
        },
    );
    Sse::new(stream).keep_alive(KeepAlive::new().interval(keepalive).text("keep-alive")).into_response()
}

fn chat_prompt(s: &AppState, body: &Value) -> Result<(Vec<u32>, bool), ApiError> {
    crate::field_policy::validate_model(body, &s.model_name, &s.model_aliases)?;
    api::parse_params_with(body, &s.defaults, &s.reasoning_fields())?;
    crate::field_policy::validate_chat_with(body, &s.request_budget.limits, &s.reasoning_caps.caps)?;
    let messages = body.get("messages").ok_or_else(|| ApiError::bad("messages is required"))?;
    if !messages.is_array() {
        return Err(ApiError::bad("messages must be a list"));
    }
    s.format.check_media(messages, s.vision.is_some())?;
    let kwargs = s.reasoning(body)?.kwargs;
    let rendered = s.format.render_bounded(messages, &kwargs, s.request_budget.limits.rendered_bytes).map_err(|e| match e {
        TemplateError::Limit => crate::budget::too_large("rendered prompt byte limit exceeded"),
        TemplateError::Raised(m) => ApiError::bad(m),
        TemplateError::Render(m) => ApiError::bad(format!("chat template failed: {m}")),
    })?;
    let ids = s.tok.encode_bounded(&rendered.text, s.request_budget.limits.rendered_bytes)?;
    Ok((ids, rendered.starts_in_reasoning))
}

/// What a failed encode of image `r` means to the client: the image memory is all in use (503, try again), the device failed (500), or the image
/// is one the encoder cannot take (400).
fn encode_error(r: &crate::vision::ImageRef, e: gb_model::error::ModelError) -> ApiError {
    use gb_model::error::ModelError;
    let at = format!("messages[{}].content[{}]", r.message, r.part);
    match e {
        ModelError::Busy(m) => ApiError::status(503, "server_overloaded", format!("{at}: {m}")).retry_after(5),
        ModelError::Cuda(_) | ModelError::StateLost(_) => {
            ApiError::status(500, "server_error", format!("{at}: image encoding failed: {e}"))
        }
        e => ApiError::bad(format!("{at}: image encoding failed: {e}")),
    }
}

/// The chat prompt with its images: the template's text, one encoder pass per image (decode, resize and encode run on the blocking
/// pool, serialized by the encoder), and every `<|image_pad|>` expanded to the image's token count.
async fn prepare_text(
    s: &Arc<AppState>,
    body: &Arc<Value>,
    lease: Arc<crate::budget::RequestLease>,
    work: &crate::preparation::RequestWork,
) -> Result<(Vec<u32>, bool), ApiError> {
    work.check()?;
    let state = s.clone();
    let inputs = (body.clone(), lease);
    let work = work.clone();
    tokio::task::spawn_blocking(move || {
        let _owner = &inputs; // Capture the WHOLE owner; body drops before its permit.
        work.check()?;
        let result = chat_prompt(&state, &inputs.0);
        work.check()?;
        result
    })
    .await
    .map_err(|_| ApiError::status(500, "server_error", "prompt preparation failed"))?
}

async fn chat_prompt_mm(
    s: &Arc<AppState>,
    body: &Arc<Value>,
    lease: Arc<crate::budget::RequestLease>,
    work: &crate::preparation::RequestWork,
) -> Result<(Vec<u32>, bool, Vec<gb_engine::ImageSpan>), ApiError> {
    let Some(v) = s.vision.as_ref() else {
        let (ids, reasoning) = prepare_text(s, body, lease, work).await?;
        return Ok((ids, reasoning, Vec::new()));
    };
    let messages = body.get("messages").unwrap_or(&Value::Null);
    // a video part is a request of its own kind (frames, tubelets, the labelled prompt): the format that takes one says so
    if let Some(vf) = s.format.video().filter(|vf| vf.has_video(messages)) {
        return video_prompt(s, v, vf, body, lease, work).await;
    }
    let retained = crate::vision::image_input_reserve(messages, &v.limits, |p| s.format.is_image_part(p))?;
    if retained.is_none() {
        let (ids, reasoning) = prepare_text(s, body, lease, work).await?;
        return Ok((ids, reasoning, Vec::new()));
    }
    // Acquire before allocating URL copies/rendering/decoder input; waiting requests cannot allocate an uncharged ImageRef backlog.
    let permit =
        v.gate.acquire(crate::preparation::image_work_bytes(v.limits.max_tokens_per_image)?, retained.unwrap(), work).await?;
    work.check()?;
    let refs = s.format.extract_images(messages, &v.limits)?;
    let state = s.clone();
    let render_body = body.clone();
    let work = work.clone();
    let ingress = v.ingress.clone();
    let gate = v.gate.clone();
    let rt = tokio::runtime::Handle::current();
    let svc = v.slot.get().cloned().ok_or_else(|| ApiError::bad("the image encoder is not available on this server"))?;
    let (limits, pp) = (v.limits, v.pp.clone());
    // everything that changes what the model sees of an image is in the key: its resized pixels and size (`Preprocessed::pixel_hash`) and the
    // preprocessing's own identity (`ImagePreprocessor::id`, FNV-1a here)
    let pp_id = pp.id().bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
    let job = crate::preparation::PreparationJob::new((render_body, refs, lease), permit);
    let (ids, in_reasoning, encoded) = tokio::task::spawn_blocking(move || {
        // Job field order destroys input owners before returning byte/concurrency/backlog credit, including error and unwind.
        let result = (|| {
            let (render_body, refs, _request_lease) = job.inputs(); // whole-job capture prevents Rust 2021 disjoint-field lease loss
            work.check()?;
            let (ids, in_reasoning) = chat_prompt(&state, render_body)?;
            work.check()?;
            let mut out = Vec::with_capacity(refs.len());
            let mut total = 0usize;
            for r in refs {
                work.check()?;
                let t0 = std::time::Instant::now();
                let bytes = if r.url.starts_with("data:") {
                    if r.url.len() > limits.max_bytes.saturating_div(3).saturating_mul(4).saturating_add(64) {
                        return Err(ApiError::status(413, "invalid_request_error", "image exceeds encoded byte limit"));
                    }
                    gb_vision::decode::data_uri_bytes(&r.url).map_err(|_| {
                        ApiError::bad(format!("messages[{}].content[{}]: invalid image data URI", r.message, r.part))
                    })?
                } else {
                    rt.block_on(ingress.fetch(&r.url, &work)).map_err(|mut e| {
                        e.message = format!("messages[{}].content[{}]: {}", r.message, r.part, e.message);
                        e
                    })?
                };
                total = total.checked_add(bytes.len()).ok_or_else(|| ApiError::bad("image aggregate size overflow"))?;
                if total > limits.max_total_bytes {
                    return Err(ApiError::status(
                        413,
                        "invalid_request_error",
                        format!(
                            "messages[{}].content[{}]: aggregate image bytes exceed {}",
                            r.message, r.part, limits.max_total_bytes
                        ),
                    ));
                }
                work.check()?;
                let pre = crate::vision::prepare_bytes(r, &bytes, &limits, &*pp, Some(&work))?;
                let t1 = std::time::Instant::now();
                work.check()?;
                let enc = svc
                    .encode_checked(&pre, || {
                        work.check().map_err(|_| gb_model::error::ModelError::Request("image work cancelled".into()))
                    })
                    .map_err(|e| work.check().err().unwrap_or_else(|| encode_error(r, e)))?;
                work.check()?;
                gate.processed.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                "rexl3 event=image_encoded message={} part={} tokens={} grid={}x{} decode_preprocess_ms={:.1} encode_ms={:.1}",
                r.message,
                r.part,
                enc.tokens,
                pre.grid.1,
                pre.grid.2,
                t1.duration_since(t0).as_secs_f64() * 1e3,
                t1.elapsed().as_secs_f64() * 1e3
            );
                let key = gb_sched::image_content_hash(pre.pixel_hash, pre.size.0 as u32, pre.size.1 as u32, pp_id);
                out.push((enc, key));
            }
            Ok::<_, ApiError>((ids, in_reasoning, out))
        })();
        if result.as_ref().is_err_and(|e| e.status == 499) {
            gate.cancelled.fetch_add(1, Ordering::Relaxed);
        }
        result
    })
    .await
    .map_err(|_| {
        eprintln!("rexl3 level=error event=image_worker_join_error");
        ApiError::status(500, "server_error", "internal image preparation worker failed (InternalJoinError)")
    })??;
    let refs: Vec<crate::chat_format::EncodedRef<'_>> = encoded
        .iter()
        .map(|(e, _)| crate::chat_format::EncodedRef { tokens: e.tokens, merged: e.merged, features: &*e.features })
        .collect();
    let (ids, placed) = s.format.place_images(&s.tok, &ids, &refs)?;
    drop(refs);
    let images = placed
        .iter()
        .zip(encoded)
        .map(|(sp, (e, key))| gb_engine::ImageSpan {
            start: sp.start,
            tokens: sp.tokens,
            merged: sp.merged,
            features: e.features,
            key: key.max(1), // 0 is "unknown": a key that hashes to it still names this image
        })
        .collect();
    Ok((ids, in_reasoning, images))
}

/// The prompt of a request with a video part (`ChatFormat::video`): the preparation gate is charged for the frames before any is copied, then
/// on the blocking pool the frames are fetched (data URIs, or the HTTPS ingress under the image policy), decoded and preprocessed one at a
/// time, the tubelets encoded (the retention mask, when pruning is on, computed on the host from the downloaded features) and the prompt
/// expanded: the processor's labels and one run of placeholders per kept tubelet, one span each.
async fn video_prompt(
    s: &Arc<AppState>,
    v: &VisionHandle,
    vf: &dyn crate::formats::nemotron_omni::prep::VideoIngress,
    body: &Arc<Value>,
    lease: Arc<crate::budget::RequestLease>,
    work: &crate::preparation::RequestWork,
) -> Result<(Vec<u32>, bool, Vec<gb_engine::ImageSpan>), ApiError> {
    let messages = body.get("messages").unwrap_or(&Value::Null);
    // sized from the request alone, before any frame URL is copied
    let retained = vf.reserve(messages, &v.limits)?.unwrap_or(0);
    let permit = v.gate.acquire(vf.work_bytes(&v.limits)?, retained, work).await?;
    work.check()?;
    let svc = v.slot.get().cloned().ok_or_else(|| ApiError::bad("the image encoder is not available on this server"))?;
    let (state, work, ingress, gate, limits) = (s.clone(), work.clone(), v.ingress.clone(), v.gate.clone(), v.limits);
    let rt = tokio::runtime::Handle::current();
    let job = crate::preparation::PreparationJob::new((body.clone(), lease), permit);
    tokio::task::spawn_blocking(move || {
        // Job field order destroys input owners before returning byte/concurrency/backlog credit, including error and unwind.
        let result = (|| {
            let (body, _request_lease) = job.inputs();
            work.check()?;
            let budget = &state.request_budget.limits;
            crate::field_policy::validate_chat_with(body, budget, &state.reasoning_caps.caps)?;
            let messages = body.get("messages").unwrap_or(&Value::Null);
            state.format.check_media(messages, state.vision.is_some())?;
            let kwargs = state.reasoning(body)?.kwargs;
            let vf = state.format.video().ok_or_else(|| ApiError::bad("video input is not enabled on this server"))?;
            let encoder = crate::formats::nemotron_omni::prep::ServiceVideoEncoder::new(&svc, Some(&work));
            let mut fetch = |r: &crate::vision::ImageRef| -> Result<Vec<u8>, ApiError> {
                if r.url.starts_with("data:") {
                    crate::formats::nemotron_omni::prep::data_uri(r, &limits)
                } else {
                    rt.block_on(ingress.fetch(&r.url, &work)).map_err(|mut e| {
                        e.message = format!("messages[{}].content[{}]: {}", r.message, r.part, e.message);
                        e
                    })
                }
            };
            let t0 = std::time::Instant::now();
            let ing = if vf.takes_frames() {
                // a format whose video is frames through its image encoder (MiniCPM-V): the service, not a tubelet encoder
                vf.ingest_frames(&state.tok, messages, &kwargs, &limits, budget.rendered_bytes, &svc, Some(&work), &mut fetch)?
            } else {
                vf.ingest(&state.tok, messages, &kwargs, &limits, budget.rendered_bytes, &encoder, Some(&work), &mut fetch)?
            };
            work.check()?;
            gate.processed.fetch_add(1, Ordering::Relaxed);
            let kept: usize = ing.images.iter().map(|i| i.tokens).sum();
            eprintln!(
                "rexl3 event=video_encoded frames={} tubelets={} tile={}x{} tokens_per_tubelet={} kept_tokens={} prompt_tokens={} prepare_ms={:.1} encode_ms={:.1}",
                ing.video.as_ref().map_or(0, |v| v.plan.frames),
                ing.video.as_ref().map_or(0, |v| v.plan.tubelets),
                ing.video.as_ref().map_or(0, |v| v.plan.tile.w_patches),
                ing.video.as_ref().map_or(0, |v| v.plan.tile.h_patches),
                ing.video.as_ref().map_or(0, |v| v.plan.tokens_per_tubelet),
                kept,
                ing.ids.len(),
                (t0.elapsed().as_secs_f64() * 1e3) - encoder.encode_us.load(Ordering::Relaxed) as f64 / 1e3,
                encoder.encode_us.load(Ordering::Relaxed) as f64 / 1e3,
            );
            Ok::<_, ApiError>((ing.ids, ing.starts_in_reasoning, ing.images))
        })();
        if result.as_ref().is_err_and(|e| e.status == 499) {
            gate.cancelled.fetch_add(1, Ordering::Relaxed);
        }
        result
    })
    .await
    .map_err(|_| {
        eprintln!("rexl3 level=error event=video_worker_join_error");
        ApiError::status(500, "server_error", "internal video preparation worker failed (InternalJoinError)")
    })?
}

/// `POST /tokenize {"text": "..."}`: the pack's tokenizer, exactly as a prompt would be encoded (`add_bos` defaults to
/// false and `encode_special_tokens` to true, so a rendered chat prompt round-trips). The oracle's other half: both sides
/// tokenize the same strings, and `prompt_logprobs` then runs the same ids.
async fn tokenize(
    State(s): State<Arc<AppState>>,
    Extension(RequestDeadline(deadline, _)): Extension<RequestDeadline>,
    request: ApiJson,
) -> Response {
    let body = &request.0;
    if let Err(e) = crate::field_policy::validate_model(body, &s.model_name, &s.model_aliases) {
        return err(e);
    }
    let Some(text) = body.get("text").or_else(|| body.get("prompt")).and_then(Value::as_str) else {
        return err(ApiError::bad("text must be a string"));
    };
    if text.len() > s.request_budget.limits.rendered_bytes {
        return err(crate::budget::too_large("tokenizer input exceeds rendered prompt byte limit"));
    }
    // `add_bos` is accepted for compatibility and is a no-op: the pack's chat template already writes its BOS, and a
    // plain text encode that added another one would not match what a prompt of that text becomes.
    let _ = match crate::field_policy::boolean(body, "add_bos", false) {
        Ok(b) => b,
        Err(e) => return err(e),
    };
    let state = s.clone();
    let text = text.to_string();
    let ids = match tokio::task::spawn_blocking(move || {
        if Instant::now() >= deadline {
            return Err(refusal_error(Refusal::RequestTimeout, state.max_model_len));
        }
        crate::tokenizer::off_workers(|| state.tok.encode_bounded(&text, state.request_budget.limits.rendered_bytes))
    })
    .await
    {
        Ok(Ok(i)) => i,
        Ok(Err(e)) => return err(e),
        Err(_) => return err(ApiError::status(500, "server_error", "tokenizer preparation failed")),
    };
    Json(json!({"tokens": ids, "count": ids.len(), "model": s.model_name})).into_response()
}

async fn render(
    State(s): State<Arc<AppState>>,
    Extension(RequestDeadline(deadline, _)): Extension<RequestDeadline>,
    request: ApiJson,
) -> Response {
    let work = crate::preparation::RequestWork::new(deadline.saturating_duration_since(Instant::now()));
    let _cancel = crate::preparation::CancelOnDrop(work.clone());
    match prepare_text(&s, &request.0, request.1.clone(), &work).await {
        Ok((ids, _)) => Json(json!({"token_ids": ids, "count": ids.len()})).into_response(),
        Err(e) => err(e),
    }
}

fn chunk(id: &str, model: &str, created: u64, delta: Value, finish: Value, extra: Option<(&str, Value)>) -> Value {
    let mut choice = json!({"index": 0, "delta": delta, "finish_reason": finish});
    if let Some((k, v)) = extra {
        choice[k] = v;
    }
    json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "system_fingerprint": fp(), "choices": [choice]})
}

/// `response` with the note of `reasoning_controls::thinking_ignored` as `X-Rexl3-Note`, when there is one.
fn with_note(mut response: Response, note: Option<String>) -> Response {
    if let Some(v) = note.and_then(|n| axum::http::HeaderValue::from_str(&n).ok()) {
        response.headers_mut().insert("x-rexl3-note", v);
    }
    response
}

async fn chat(
    State(s): State<Arc<AppState>>,
    Extension(RequestDeadline(deadline, failure_hook)): Extension<RequestDeadline>,
    Extension(arrival): Extension<Arrival>,
    request: ApiJson,
) -> Response {
    // Keep Value and its lease together on every early refusal: tuple fields drop in declaration order (Value before lease).
    let body = &request.0;
    let lease = request.1.clone();
    let model = match crate::field_policy::validate_model(body, &s.model_name, &s.model_aliases) {
        Ok(m) => m,
        Err(e) => return err(e),
    };
    let p = match api::parse_params_with(body, &s.defaults, &s.reasoning_fields()) {
        Ok(p) => p,
        Err(e) => return err(e),
    };
    if p.prompt_logprobs > 0 {
        return err(crate::field_policy::bad(
            "logprobs",
            "logprobs/prompt_logprobs are served by /v1/completions (a teacher-forced prompt readout), not by chat completions",
        ));
    }
    let timeout = s
        .vision
        .as_ref()
        .map_or(Duration::from_secs(1800), |v| v.gate.limits.deadline)
        .min(deadline.saturating_duration_since(Instant::now()));
    let work = crate::preparation::RequestWork::new(timeout);
    let work = s.vision.as_ref().map_or(work.clone(), |v| work.with_shutdown(v.gate.shutdown_token()));
    let _cancel = crate::preparation::CancelOnDrop(work.clone());
    let preparation = tokio::time::timeout(work.remaining(), chat_prompt_mm(&s, body, lease.clone(), &work)).await;
    let prepared = preparation.unwrap_or_else(|_| {
        work.cancel();
        if let Some(v) = &s.vision {
            v.gate.timed_out.fetch_add(1, Ordering::Relaxed);
        }
        Err(crate::preparation::timeout_error())
    });
    let (ids, in_reasoning, images) = match prepared {
        Ok(x) => x,
        Err(e) => return err(e),
    };
    let note = crate::reasoning_controls::thinking_ignored(body, &s.reasoning_caps.caps, in_reasoning);
    // the reasoning budget (already validated while the prompt was prepared): it caps a reply that starts inside a thought
    let reasoning_budget = match crate::reasoning_controls::budget(body, &s.reasoning_caps.caps, s.default_thinking) {
        // the descriptor offers a budget only with a one-token closing marker (`reasoning_caps::finish`), so the id is there
        Ok(Some(after)) if in_reasoning => match s.reasoning_caps.close_id {
            Some(close) => Some(gb_engine::ReasoningBudget { after, close }),
            None => {
                return err(crate::field_policy::bad(
                    "max_reasoning_tokens",
                    s.reasoning_caps.budget_refusal.clone().unwrap_or_else(|| "this model offers no reasoning budget".into()),
                ))
            }
        },
        Ok(_) => None,
        Err(e) => return err(e),
    };
    let tool_schemas = crate::tools::ToolSchemas::from_request(body);
    drop(request);
    drop(lease);
    let Started { mut rx, plen, guard, failure_hook } = match start(
        &s,
        (arrival, RequestDeadline(deadline, failure_hook)),
        ids,
        &p,
        in_reasoning,
        images,
        tool_schemas,
        reasoning_budget,
    )
    .await
    {
        Ok(x) => x,
        Err(e) => return err(e),
    };
    let id = format!("chatcmpl-{:x}", rand_id());
    let created = api::now();
    if p.stream {
        let include_usage = p.include_usage;
        let want_ids = p.return_token_ids;
        let (etx, erx) = channel_with_failure::<WireEvent>(Some(failure_hook));
        let stall = s.handle.write_stall;
        tokio::spawn(async move {
            let send = |v: Value| {
                let etx = etx.clone();
                async move { etx.send_wait(WireEvent(v.to_string(), None), deadline, stall).await.is_ok() }
            };
            if !send(chunk(&id, &model, created, json!({"role": "assistant", "content": ""}), Value::Null, None)).await {
                return;
            }
            let mut tc_index = 0usize;
            while let Some(o) = rx.recv().await {
                match o {
                    Out::Text { reasoning, content, ids, calls } => {
                        let extra = || want_ids.then(|| ("token_ids", json!(ids)));
                        for c in &calls {
                            let delta = json!({"tool_calls": [{"index": tc_index, "id": format!("call_{:x}", rand_id()), "type": "function",
                                "function": {"name": c.name, "arguments": c.arguments}}]});
                            tc_index += 1;
                            if !send(chunk(&id, &model, created, delta, Value::Null, None)).await {
                                return;
                            }
                        }
                        let mut first_extra = true;
                        if !reasoning.is_empty() {
                            let e = if first_extra {
                                first_extra = false;
                                extra()
                            } else {
                                None
                            };
                            if !send(chunk(
                                &id,
                                &model,
                                created,
                                json!({"reasoning_content": reasoning, "reasoning": reasoning}),
                                Value::Null,
                                e,
                            ))
                            .await
                            {
                                return;
                            }
                        }
                        if !content.is_empty() {
                            let e = if first_extra {
                                first_extra = false;
                                extra()
                            } else {
                                None
                            };
                            if !send(chunk(&id, &model, created, json!({"content": content}), Value::Null, e)).await {
                                return;
                            }
                        }
                        if first_extra
                            && want_ids
                            && !ids.is_empty()
                            && !send(chunk(&id, &model, created, json!({}), Value::Null, extra())).await
                        {
                            return;
                        }
                    }
                    Out::Done { finish, usage, error } => {
                        let failed = error.is_some();
                        // a reply that was cut short ends with an error event and no finish chunk: never as a normal stop, and the
                        // stream then CLOSES with no `[DONE]` (the OpenAI convention for an errored stream): a client that takes
                        // `[DONE]` for completion must not read a short reply as a whole one. The event is the stream's LAST event
                        // (no usage chunk follows it, `include_usage` or not) and carries the terminal mark (`Some(true)`), so the
                        // outcome counter still says "error".
                        if let Some(e) = error {
                            let _ = etx.send_wait(WireEvent(error_event(&e).to_string(), Some(true)), deadline, stall).await;
                        } else {
                            send(chunk(&id, &model, created, json!({}), json!(finish), None)).await;
                        }
                        // no usage chunk after an error event: it is the last event of the stream
                        if include_usage && !failed {
                            send(
                                json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "system_fingerprint": fp(), "choices": [], "usage": usage}),
                            ).await;
                        }
                        if !failed {
                            let _ = etx.send_wait(WireEvent("[DONE]".into(), Some(false)), deadline, stall).await;
                        }
                        return;
                    }
                }
            }
        });
        return with_note(sse_response(erx, guard, s.sse_keepalive, deadline, s.handle.metrics.clone()), note);
    }
    let (mut reasoning, mut content, mut ids_all) = (String::new(), String::new(), Vec::new());
    let mut tool_calls: Vec<Value> = Vec::new();
    while let Some(o) = rx.recv().await {
        match o {
            Out::Text { reasoning: r, content: c, ids, calls } => {
                reasoning.push_str(&r);
                content.push_str(&c);
                ids_all.extend(ids);
                for call in calls {
                    tool_calls.push(json!({"id": format!("call_{:x}", rand_id()), "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments}}));
                }
            }
            Out::Done { finish, usage, error } => {
                if let Some(e) = error {
                    return err(seq_error(&e));
                }
                let mut message = json!({"role": "assistant", "content": content});
                if !tool_calls.is_empty() {
                    message["tool_calls"] = json!(tool_calls);
                    if content.trim().is_empty() {
                        message["content"] = Value::Null;
                    }
                }
                if !reasoning.is_empty() || in_reasoning {
                    message["reasoning_content"] = json!(reasoning);
                    message["reasoning"] = json!(reasoning);
                }
                let mut choice = json!({"index": 0, "message": message, "finish_reason": finish});
                if p.return_token_ids {
                    choice["token_ids"] = json!(ids_all);
                }
                return with_note(
                    Json(
                        json!({"id": id, "object": "chat.completion", "created": created, "model": model, "system_fingerprint": fp(),
                                   "choices": [choice], "usage": usage, "prompt_tokens_count": plen}),
                    )
                    .into_response(),
                    note,
                );
            }
        }
    }
    err(ApiError::status(500, "server_error", "generation ended without a result"))
}

/// The log-sum-exp of a whole logits row (the log-softmax's normaliser); a non-finite or empty row is refused.
fn row_logsumexp(row: &[f32]) -> Result<f64, ApiError> {
    if row.is_empty() || row.iter().any(|v| !v.is_finite()) {
        return Err(ApiError::status(500, "server_error", "the engine returned a non-finite logits row"));
    }
    Ok(crate::decisions::logsumexp_full_row(row))
}

/// One scored prompt position: `row` is the logits after the prompt's tokens before `token`, so it predicts `token`.
/// Returns `token`'s exact log-softmax under `row` (whether or not it is among the top `k`) and the row's top-`k`
/// `(logprob, id)` pairs, both from one normaliser, so a token in the top `k` has exactly its top-`k` value.
fn scored_position(row: &[f32], token: u32, k: usize) -> Result<(f32, Vec<(f32, u32)>), ApiError> {
    let lse = row_logsumexp(row)?;
    let own = match row.get(token as usize) {
        Some(&v) => (v as f64 - lse) as f32,
        None => {
            return Err(ApiError::status(
                500,
                "server_error",
                format!("prompt token {token} is outside the engine's {}-entry logits row", row.len()),
            ))
        }
    };
    Ok((own, top_k_under(row, lse, k)))
}

/// The top-`k` `(logprob, id)` pairs of one logits row under its log-softmax normaliser `lse` (from [`row_logsumexp`]: f64
/// log-sum-exp with compensated summation over the full vocabulary). Descending by logprob.
fn top_k_under(row: &[f32], lse: f64, k: usize) -> Vec<(f32, u32)> {
    let mut lps: Vec<(f32, u32)> = row.iter().enumerate().map(|(t, &v)| ((v as f64 - lse) as f32, t as u32)).collect();
    let k = k.min(lps.len());
    if k < lps.len() {
        lps.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
        lps.truncate(k);
    }
    lps.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    lps
}

/// The `logprobs` object and the `prompt_logprobs` entries of a teacher-forced readout of `ids`. `rows[i]` is the logits
/// row after `ids[..=i]`: it predicts `ids[i + 1]`. So position `i >= 1` reports `ids[i]` with its exact log-softmax under
/// `rows[i - 1]` and that row's top `k`; position 0 has no conditioning prefix and is null; the last row predicts past the
/// prompt and is not reported (vLLM's prompt-logprobs convention). Every array has one entry per prompt token.
fn prompt_logprobs_body(
    rows: &[Vec<f32>],
    ids: &[u32],
    k: usize,
    piece: impl Fn(u32) -> String,
) -> Result<(Value, Vec<Value>), ApiError> {
    if rows.len() != ids.len() {
        return Err(ApiError::status(
            500,
            "server_error",
            format!("the engine returned {} logits rows for {} prompt tokens", rows.len(), ids.len()),
        ));
    }
    let n = ids.len();
    let mut tokens: Vec<Value> = Vec::with_capacity(n);
    let mut token_ids: Vec<Value> = Vec::with_capacity(n);
    let mut token_logprobs: Vec<Value> = Vec::with_capacity(n);
    let mut top_logprobs: Vec<Value> = Vec::with_capacity(n);
    let mut offsets: Vec<Value> = Vec::with_capacity(n);
    let mut entries: Vec<Value> = Vec::with_capacity(n);
    let mut offset = 0usize;
    for (i, &tok) in ids.iter().enumerate() {
        offsets.push(json!(offset));
        let text = piece(tok);
        offset += text.len();
        if i == 0 {
            tokens.push(Value::Null);
            token_ids.push(Value::Null);
            token_logprobs.push(Value::Null);
            top_logprobs.push(Value::Null);
            entries.push(json!({"position": 0, "token_id": null, "logprob": null, "top_ids": [], "top_logprobs": []}));
            continue;
        }
        let (own, lps) = scored_position(&rows[i - 1], tok, k)?;
        let top_ids: Vec<u32> = lps.iter().map(|x| x.1).collect();
        let top_vals: Vec<f32> = lps.iter().map(|x| x.0).collect();
        tokens.push(json!(text));
        token_ids.push(json!(tok));
        token_logprobs.push(json!(own));
        top_logprobs.push(Value::Object(top_ids.iter().zip(top_vals.iter()).map(|(t, v)| (t.to_string(), json!(v))).collect()));
        entries.push(json!({"position": i, "token_id": tok, "logprob": own, "top_ids": top_ids, "top_logprobs": top_vals}));
    }
    let logprobs = json!({
        "tokens": tokens,
        "token_ids": token_ids,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": offsets,
    });
    Ok((logprobs, entries))
}

/// `prompt_logprobs: k` on /v1/completions: one teacher-forced pass over the whole prompt, then each position's top-`k`
/// token ids with their EXACT log-softmax over the full vocabulary (f64 log-sum-exp with compensated summation; the model's
/// own f16 head, not the sampler's candidate list). Nothing is sampled and no KV is kept. vLLM's prompt-logprobs convention:
/// position `i` scores prompt token `i` from the logits after tokens `0..i` ([`prompt_logprobs_body`]); the first position
/// has no conditioning prefix, so its entry is null, and a prompt of one token has no scored position.
async fn prompt_logprobs(
    s: &Arc<AppState>,
    ids: Vec<u32>,
    model: &str,
    k: usize,
    deadline: Instant,
    max_model_len: usize,
) -> Response {
    let id = format!("cmpl-{:x}", rand_id());
    let created = api::now();
    let remaining = deadline.saturating_duration_since(Instant::now());
    let (guard, rx) = match s
        .handle
        .score_with_timeout(gb_engine::score::ScoreRequest { prompt: ids.clone(), cache: false, all_rows: true }, remaining)
        .await
    {
        Ok(x) => x,
        Err(e) => return err(refusal_error(e, max_model_len)),
    };
    let row = match tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), rx).await {
        Ok(Ok(Ok(r))) => r,
        Ok(Ok(Err(e))) => return err(seq_error(&e)),
        Ok(Err(_)) => return err(ApiError::status(503, "server_error", "the score engine stopped")),
        Err(_) => {
            return err(ApiError {
                code: Some("request_timeout"),
                ..ApiError::status(504, "request_timeout", "prompt_logprobs deadline exceeded")
            })
        }
    };
    drop(guard);
    let (logprobs, entries) = match prompt_logprobs_body(&row.logits_rows, &ids, k, |t| s.tok.decode(&[t], true)) {
        Ok(x) => x,
        Err(e) => return err(e),
    };
    let n = ids.len();
    Json(json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "system_fingerprint": fp(),
        "choices": [{
            "index": 0,
            "text": "",
            "finish_reason": "stop",
            "logprobs": logprobs,
        }],
        "prompt_logprobs": entries,
        "usage": {"prompt_tokens": n, "completion_tokens": 0, "total_tokens": n},
    }))
    .into_response()
}

async fn completions(
    State(s): State<Arc<AppState>>,
    Extension(RequestDeadline(deadline, failure_hook)): Extension<RequestDeadline>,
    Extension(arrival): Extension<Arrival>,
    request: ApiJson,
) -> Response {
    let body = &request.0;
    let lease = request.1.clone();
    let model = match crate::field_policy::validate_model(body, &s.model_name, &s.model_aliases) {
        Ok(m) => m,
        Err(e) => return err(e),
    };
    if let Err(e) = crate::field_policy::validate_completions_with(body, &s.reasoning_fields()) {
        return err(e);
    }
    let p = match api::parse_params_with(body, &s.defaults, &s.reasoning_fields()) {
        Ok(p) => p,
        Err(e) => return err(e),
    };
    let ids: Vec<u32> = match body.get("prompt") {
        Some(Value::String(t)) => {
            if t.len() > s.request_budget.limits.rendered_bytes {
                return err(crate::budget::too_large("tokenizer input exceeds rendered prompt byte limit"));
            }
            let (state, inputs) = (s.clone(), (request.0.clone(), lease.clone()));
            match tokio::task::spawn_blocking(move || {
                let _owner = &inputs;
                if Instant::now() >= deadline {
                    return Err(refusal_error(Refusal::RequestTimeout, state.max_model_len));
                }
                let text = inputs.0.get("prompt").and_then(Value::as_str).expect("validated string prompt");
                let ids =
                    crate::tokenizer::off_workers(|| state.tok.encode_bounded(text, state.request_budget.limits.rendered_bytes))?;
                if Instant::now() >= deadline {
                    return Err(refusal_error(Refusal::RequestTimeout, state.max_model_len));
                }
                Ok::<_, ApiError>(ids)
            })
            .await
            {
                Ok(Ok(i)) => i,
                Ok(Err(e)) => return err(e),
                Err(_) => return err(ApiError::status(500, "server_error", "tokenizer preparation failed")),
            }
        }
        Some(Value::Array(a)) => match api::token_prompt(a) {
            Some(ids) => ids,
            None => {
                return err(ApiError::bad(
                    "prompt must be a string or a non-empty list of token ids (integers from 0 to 4294967295)",
                ))
            }
        },
        _ => return err(ApiError::bad("prompt must be a string or a token-id list")),
    };
    drop(request);
    drop(lease);
    if p.prompt_logprobs > 0 {
        return prompt_logprobs(&s, ids, &model, p.prompt_logprobs, deadline, s.max_model_len).await;
    }
    let Started { mut rx, guard, failure_hook, .. } =
        match start(&s, (arrival, RequestDeadline(deadline, failure_hook)), ids, &p, false, Vec::new(), None, None).await {
            Ok(x) => x,
            Err(e) => return err(e.for_field("prompt")),
        };
    let id = format!("cmpl-{:x}", rand_id());
    let created = api::now();
    if p.stream {
        let include_usage = p.include_usage;
        let (etx, erx) = channel_with_failure::<WireEvent>(Some(failure_hook));
        let stall = s.handle.write_stall;
        tokio::spawn(async move {
            let send = |v: Value| {
                let etx = etx.clone();
                async move { etx.send_wait(WireEvent(v.to_string(), None), deadline, stall).await.is_ok() }
            };
            while let Some(o) = rx.recv().await {
                match o {
                    Out::Text { reasoning, content, .. } => {
                        let text = format!("{reasoning}{content}");
                        if !text.is_empty()
                            && !send(
                                json!({"id": id, "object": "text_completion", "created": created, "model": model, "system_fingerprint": fp(),
                                            "choices": [{"index": 0, "text": text, "finish_reason": null}]}),
                            ).await
                        {
                            return;
                        }
                    }
                    Out::Done { finish, usage, error } => {
                        let failed = error.is_some();
                        // an errored stream ends with its error event and then closes: no `[DONE]` (see the chat stream above)
                        if let Some(e) = error {
                            let _ = etx.send_wait(WireEvent(error_event(&e).to_string(), Some(true)), deadline, stall).await;
                        } else {
                            send(
                                json!({"id": id, "object": "text_completion", "created": created, "model": model, "system_fingerprint": fp(),
                                        "choices": [{"index": 0, "text": "", "finish_reason": finish}]}),
                            ).await;
                        }
                        if include_usage && !failed {
                            send(
                                json!({"id": id, "object": "text_completion", "created": created, "model": model, "system_fingerprint": fp(), "choices": [], "usage": usage}),
                            ).await;
                        }
                        if !failed {
                            let _ = etx.send_wait(WireEvent("[DONE]".into(), Some(false)), deadline, stall).await;
                        }
                        return;
                    }
                }
            }
        });
        return sse_response(erx, guard, s.sse_keepalive, deadline, s.handle.metrics.clone());
    }
    let mut text = String::new();
    while let Some(o) = rx.recv().await {
        match o {
            Out::Text { reasoning, content, .. } => {
                text.push_str(&reasoning);
                text.push_str(&content);
            }
            Out::Done { finish, usage, error } => {
                if let Some(e) = error {
                    return err(seq_error(&e));
                }
                return Json(
                    json!({"id": id, "object": "text_completion", "created": created, "model": model, "system_fingerprint": fp(),
                                   "choices": [{"index": 0, "text": text, "finish_reason": finish}], "usage": usage}),
                )
                .into_response();
            }
        }
    }
    err(ApiError::status(500, "server_error", "generation ended without a result"))
}

fn rand_id() -> u64 {
    use std::sync::atomic::AtomicU64;
    static N: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    t.wrapping_mul(0x9E3779B97F4A7C15) ^ N.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {

    #[test]
    fn top_k_logprobs_is_the_exact_softmax_of_the_full_row() {
        let row = [0.0_f32, 1.0, 2.0, -1.0, 5.0];
        fn top_k_logprobs(row: &[f32], k: usize) -> Result<Vec<(f32, u32)>, ApiError> {
            row_logsumexp(row).map(|lse| top_k_under(row, lse, k))
        }
        let top = top_k_logprobs(&row, 2).unwrap();
        assert_eq!(top.iter().map(|x| x.1).collect::<Vec<_>>(), vec![4, 2], "descending by logit");
        // Z = e^0 + e^1 + e^2 + e^-1 + e^5; log p_4 = 5 - log Z and log p_2 = 2 - log Z
        let z: f64 = [0.0_f64, 1.0, 2.0, -1.0, 5.0].iter().map(|x| x.exp()).sum();
        assert!((top[0].0 as f64 - (5.0 - z.ln())).abs() < 1e-6);
        assert!((top[1].0 as f64 - (2.0 - z.ln())).abs() < 1e-6);
        assert!(top_k_logprobs(&[f32::NAN], 1).is_err(), "a non-finite row is refused");
        assert_eq!(top_k_logprobs(&row, 99).unwrap().len(), 5, "k past the vocabulary takes the whole row");
    }

    /// `rows[i]` is the logits after prompt tokens `0..=i`, so it predicts token `i + 1`: position `i` reports token `i` from
    /// `rows[i - 1]`, with its exact log-softmax even when it is outside the top `k`; position 0 is null; the last row (a
    /// prediction past the prompt) is never read.
    #[test]
    fn a_prompt_position_scores_its_token_from_the_row_before_it() {
        // vocabulary 6; row j peaks at token (j + 2) % 6, so a row read one position off shows in the top ids
        let ids = [4u32, 2, 5, 1];
        let rows: Vec<Vec<f32>> = (0..ids.len())
            .map(|j| (0..6).map(|t| if t == (j + 2) % 6 { 4.0 } else { t as f32 * 0.25 - 0.5 * j as f32 }).collect())
            .collect();
        let lsm = |row: &[f32], t: u32| {
            let z: f64 = row.iter().map(|&v| (v as f64).exp()).sum();
            row[t as usize] as f64 - z.ln()
        };
        let (lp, entries) = prompt_logprobs_body(&rows, &ids, 2, |t| format!("<{t}>")).unwrap();
        assert_eq!(entries.len(), ids.len());
        for key in ["tokens", "token_ids", "token_logprobs", "top_logprobs", "text_offset"] {
            assert_eq!(lp[key].as_array().unwrap().len(), ids.len(), "{key}: one entry per prompt token");
        }
        assert!(lp["tokens"][0].is_null() && lp["token_ids"][0].is_null() && lp["token_logprobs"][0].is_null());
        assert!(lp["top_logprobs"][0].is_null() && entries[0]["logprob"].is_null() && entries[0]["token_id"].is_null());
        assert_eq!(lp["text_offset"], json!([0, 3, 6, 9]));
        for i in 1..ids.len() {
            let e = &entries[i];
            assert_eq!((&e["position"], &e["token_id"], &lp["token_ids"][i]), (&json!(i), &json!(ids[i]), &json!(ids[i])));
            assert_eq!(lp["tokens"][i], json!(format!("<{}>", ids[i])));
            let want = lsm(&rows[i - 1], ids[i]);
            assert!((e["logprob"].as_f64().unwrap() - want).abs() < 1e-6, "position {i}: {} vs {want}", e["logprob"]);
            assert_eq!(e["logprob"], lp["token_logprobs"][i]);
            assert_eq!(e["top_ids"][0], json!((i + 1) % 6), "position {i} reads the row after token {}", i - 1);
            assert_eq!(lp["top_logprobs"][i][((i + 1) % 6).to_string()], e["top_logprobs"][0]);
        }
        // token 2 is the peak of row 0: its value is exactly its top-k value
        assert_eq!(entries[1]["logprob"], entries[1]["top_logprobs"][0]);
        // token 1 is not in row 2's top 2 (4 and 5): it gets its own, lower value, not the top-1 value
        assert!(!entries[3]["top_ids"].as_array().unwrap().contains(&json!(1)));
        assert!(entries[3]["logprob"].as_f64().unwrap() < entries[3]["top_logprobs"][1].as_f64().unwrap());
        // the last row is never read: another (even a non-finite) last row gives the same body
        let mut other = rows.clone();
        other[3] = vec![f32::NAN; 6];
        assert_eq!(prompt_logprobs_body(&other, &ids, 2, |t| format!("<{t}>")).unwrap(), (lp, entries));
        // a one-token prompt has one null position and reads no row
        let (one, e1) = prompt_logprobs_body(&[vec![f32::NAN; 6]], &[3], 2, |_| "x".into()).unwrap();
        assert_eq!((one["token_logprobs"].clone(), e1.len()), (json!([null]), 1));
    }

    #[test]
    fn a_prompt_readout_refuses_rows_it_cannot_score() {
        let ids = [1u32, 2, 3];
        let good = vec![vec![0.0f32, 1.0, 2.0, 3.0]; 3];
        assert!(prompt_logprobs_body(&good[..2], &ids, 1, |_| String::new()).is_err(), "one row per prompt token");
        let mut nan = good.clone();
        nan[1][0] = f32::INFINITY;
        assert!(prompt_logprobs_body(&nan, &ids, 1, |_| String::new()).is_err(), "a non-finite row that scores a token");
        assert!(prompt_logprobs_body(&good, &[1, 9, 3], 1, |_| String::new()).is_err(), "a token outside the row");
    }

    #[test]
    fn a_lost_tensor_parallel_group_is_not_ready_even_when_the_engine_has_noticed() {
        assert_eq!(health_verdict(false, false, true, true, "ok"), (StatusCode::OK, "ok"));
        assert_eq!(health_verdict(false, true, true, true, "ok"), (StatusCode::SERVICE_UNAVAILABLE, "group_lost"));
        assert_eq!(health_verdict(false, false, false, true, "loading"), (StatusCode::SERVICE_UNAVAILABLE, "loading"));
        assert_eq!(
            health_verdict(true, true, true, true, "ok"),
            (StatusCode::SERVICE_UNAVAILABLE, "wedged"),
            "wedged is named first"
        );
        // a failed engine keeps its own word when the group was lost too: the group word is only for an engine that still says ok
        assert_eq!(health_verdict(false, true, false, true, "failed"), (StatusCode::SERVICE_UNAVAILABLE, "failed"));
        // a boot that is not complete (weights, warm n-gram table) is not ready even though the engine is up
        assert_eq!(health_verdict(false, false, false, false, "loading"), (StatusCode::SERVICE_UNAVAILABLE, "loading"));
        assert_eq!(health_verdict(false, false, true, false, "ok"), (StatusCode::SERVICE_UNAVAILABLE, "loading"));
    }
    use super::*;

    #[tokio::test]
    async fn unexpected_wire_producer_eof_is_terminal_error_not_silent_truncation() {
        use gb_engine::mock::{Fault, MockBackend};
        use gb_engine::{Engine, EngineConfig, NoDrafts};
        use gb_sched::{Admission, CostModel, ModelMemory};
        let (handle, thread) = crate::runtime::spawn(|| {
            Engine::new(
                MockBackend::new(256, 1, Fault::None),
                Admission::new(
                    4096,
                    0,
                    ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 0, activation_reserve: 0 },
                    1,
                )
                .unwrap(),
                EngineConfig {
                    max_rows: 64,
                    max_drafts: 0,
                    cost: CostModel::from_table(vec![(1, 1.0), (64, 2.0)]),
                    drafting: false,
                },
                Box::new(|| Box::new(NoDrafts)),
            )
        });
        let (tx, rx) = crate::flow::channel::<WireEvent>();
        drop(tx);
        let response = sse_response(
            rx,
            CancelGuard::new(handle.clone(), 999),
            Duration::from_secs(1),
            Instant::now() + Duration::from_secs(2),
            handle.metrics.clone(),
        );
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("response producer ended before a terminal"));
        assert!(text.contains("data: {\"error\""));
        assert!(!text.contains("finish_reason"));
        assert_eq!(handle.metrics.observability.sse[1].load(Ordering::Relaxed), 1);
        handle.shutdown();
        thread.join().unwrap();
    }

    #[test]
    fn stop_strings_cut_across_chunk_boundaries() {
        for chunks in
            [vec!["hello wor", "ld END and more"], vec!["hello world E", "ND and more"], vec!["hello world EN", "D", " and more"]]
        {
            let mut m = StopMatcher::new(vec!["END".into()]);
            let mut out = String::new();
            let mut hit = false;
            for c in &chunks {
                let (t, h) = m.push(c);
                out.push_str(&t);
                if h {
                    hit = true;
                    break;
                }
            }
            assert!(hit, "{chunks:?}");
            assert_eq!(out, "hello world ", "{chunks:?}");
        }
        // no match: everything comes out by the end
        let mut m = StopMatcher::new(vec!["END".into()]);
        let mut out = String::new();
        for c in ["abc EN", "x done"] {
            out.push_str(&m.push(c).0);
        }
        out.push_str(&m.finish());
        assert_eq!(out, "abc ENx done");
        // multibyte text is never split
        let mut m = StopMatcher::new(vec!["\u{2603}\u{2603}".into()]);
        let (t, _) = m.push("caf\u{e9}\u{2603}");
        assert!(t.is_char_boundary(t.len()));
    }
    #[test]
    fn an_engine_side_context_refusal_carries_the_openai_code_and_other_admission_errors_do_not() {
        let e = submit_error(SubmitError::Admit(AdmitError::ContextTooLong { tokens: 700, max: 512 }), 512);
        assert_eq!(
            (e.status, e.kind, e.code, e.param.as_deref()),
            (400, "invalid_request_error", Some("context_length_exceeded"), Some("messages"))
        );
        assert!(e.message.starts_with("This model's maximum context length is 512 tokens"), "{}", e.message);
        assert_eq!(e.clone().for_field("prompt").body()["error"]["param"], "prompt");
        let e = submit_error(SubmitError::Admit(AdmitError::TooLarge), 512);
        assert_eq!((e.code, e.param.as_deref()), (None, None));
        assert_eq!(e.body()["error"]["code"], 400);
        let e = refusal_error(Refusal::QueueFull { depth: 4 }, 512);
        assert_eq!(e.body()["error"]["code"], 429, "a full queue keeps its numeric code");
        assert_eq!(e.clone().for_field("prompt").param, None, "an error that names no field is left alone");
    }

    #[test]
    fn omitted_max_tokens_gets_the_default_cap_within_the_window() {
        assert_eq!(length_budget(200_000, None, 16_384), 16_384);
        assert_eq!(length_budget(5_000, None, 16_384), 5_000, "never past the room left");
        assert_eq!(length_budget(200_000, None, 0), 200_000, "0 keeps the whole window");
        assert_eq!(length_budget(200_000, Some(100_000), 16_384), 100_000, "an explicit value wins over the default");
        assert_eq!(length_budget(1_000, Some(100_000), 16_384), 1_000);
    }

    #[test]
    fn the_models_card_reports_the_prefix_cache_reach_from_the_boot_fact() {
        // an engine that plans a cache records its reach as a boot fact; the card reads it back, and a value that is not a number is null
        gb_engine::boot::fact("prefix_cache_reach_tokens", "265216");
        assert_eq!(prefix_cache_reach_tokens(), Some(265_216));
        gb_engine::boot::fact("prefix_cache_reach_tokens", "not a number");
        assert_eq!(prefix_cache_reach_tokens(), None);
    }
}
