//! Apache-2.0. Fixed-label HTTP and engine telemetry. No content or request identifiers in metrics.
use crate::http::AppState;
use crate::latency::Hist;
use axum::{
    extract::{Request, State},
    http::HeaderValue,
    middleware::Next,
    response::Response,
};
use std::{
    fmt::Write,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering::Relaxed},
        Arc, Mutex,
    },
    time::Instant,
};

pub const CODES: [&str; 14] = ["200", "201", "204", "400", "401", "403", "404", "405", "413", "415", "429", "500", "503", "504"];
pub const ROUTES: &[&str] = &[
    "/health",
    "/health/live",
    "/health/ready",
    "/v1/health",
    "/metrics",
    "/v1/models",
    "/v1/completions",
    "/v1/chat/completions",
    "/v1/chat/completions/render",
    "/v1/decisions",
    "other",
];
/// When a request entered the middleware, and for a generation request the mark that keeps it counted in
/// [`Observability::arriving`] until the engine has answered its submit.
#[derive(Clone)]
pub struct Arrival(pub Instant, pub ArrivalMark);

/// A generation request's place in [`Observability::arriving`]: settled once (the engine answered its submit, or the request ended
/// before submitting), whichever comes first.
#[derive(Clone, Default)]
pub struct ArrivalMark(Option<Arc<AtomicBool>>);

impl ArrivalMark {
    fn counted(o: &Observability) -> Self {
        o.arriving.fetch_add(1, Relaxed);
        Self(Some(Arc::new(AtomicBool::new(true))))
    }
    /// The request no longer arrives: it is queued, refused, or gone.
    pub fn settle(&self, o: &Observability) {
        if self.0.as_ref().is_some_and(|f| f.swap(false, Relaxed)) {
            o.arriving.fetch_sub(1, Relaxed);
        }
    }
}

/// Settles its mark when dropped (the middleware's future ended or was dropped).
struct Settle<'a>(&'a Observability, ArrivalMark);

impl Drop for Settle<'_> {
    fn drop(&mut self) {
        self.1.settle(self.0);
    }
}

#[derive(Default)]
pub struct Observability {
    pub http_entered: AtomicU64,
    /// Generation requests (`/v1/chat/completions`, `/v1/completions`) that have entered the middleware and whose submit the engine
    /// has not answered yet (parsing, templating, tokenizing, or their submit waiting in the engine's inbox). The serving loop waits
    /// its gather window for these only (`runtime::run`): a request that wakes an idle engine alone starts at once.
    pub arriving: AtomicU64,
    pub http_codes: [AtomicU64; 15],
    pub http_route_codes: [[AtomicU64; CODES.len() + 1]; ROUTES.len()],
    pub token_batches: AtomicU64,
    pub completed: AtomicU64,
    pub prefill: Mutex<Option<gb_engine::PrefillStats>>,
    pub capacity: Mutex<Option<(u64, u64)>>,
    pub admission_bytes: Mutex<Option<(u64, u64)>>,
    pub slots: AtomicU64,
    pub snapshot_at: Mutex<Option<Instant>>,
    pub sse: [AtomicU64; 3],
    pub ttft: Hist,
    pub submit_ttft: Hist,
    pub batch_gap: Hist,
    pub e2e: Hist,
    pub preprocessing: Hist,
    pub queue: Hist,
    pub prefill_slice: Hist,
    pub http_headers: Hist,
}
impl Observability {
    pub fn set_recording(&self, enabled: bool) {
        for h in [
            &self.ttft,
            &self.submit_ttft,
            &self.batch_gap,
            &self.e2e,
            &self.preprocessing,
            &self.queue,
            &self.prefill_slice,
            &self.http_headers,
        ] {
            h.set_recording(enabled);
        }
    }
    pub fn render(&self, out: &mut String) {
        let g = |a: &AtomicU64| a.load(Relaxed);
        fn one(out: &mut String, name: &str, kind: &str, help: &str, v: impl std::fmt::Display) {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}");
        }
        one(
            out,
            "rexl3_http_requests_total",
            "counter",
            "Requests entered HTTP middleware, including malformed bodies and unknown routes; not engine admissions.",
            g(&self.http_entered),
        );
        one(
            out,
            "rexl3_token_batches_total",
            "counter",
            "Nonempty engine token batches produced, including the first batch; not SSE events or client flushes.",
            g(&self.token_batches),
        );
        one(
            out,
            "rexl3_requests_completed_total",
            "counter",
            "Admitted requests successfully completed by the engine; excludes failures and cancellations.",
            g(&self.completed),
        );
        let _ = writeln!(out, "# HELP rexl3_http_responses_total HTTP response headers returned once by fixed status code, all routes; HTTP 200 is not stream success.\n# TYPE rexl3_http_responses_total counter");
        for (i, c) in CODES.iter().chain(std::iter::once(&"other")).enumerate() {
            let _ = writeln!(out, "rexl3_http_responses_total{{code=\"{c}\"}} {}", g(&self.http_codes[i]));
        }
        let _ = writeln!(out, "# HELP rexl3_http_route_responses_total Actual HTTP headers by bounded canonical route/status, includes refusals; not stream success.\n# TYPE rexl3_http_route_responses_total counter");
        for (ri, route) in ROUTES.iter().enumerate() {
            for (ci, code) in CODES.iter().chain(std::iter::once(&"other")).enumerate() {
                let _ = writeln!(
                    out,
                    "rexl3_http_route_responses_total{{route=\"{route}\",code=\"{code}\"}} {}",
                    g(&self.http_route_codes[ri][ci])
                );
            }
        }
        let _ = writeln!(out, "# HELP rexl3_sse_terminal_total Stream outcomes observed when terminal event is polled by HTTP body, or body dropped before terminal; not TCP acknowledgement.\n# TYPE rexl3_sse_terminal_total counter");
        for (i, o) in ["success", "error", "cancelled"].iter().enumerate() {
            let _ = writeln!(out, "rexl3_sse_terminal_total{{outcome=\"{o}\"}} {}", g(&self.sse[i]));
        }
        if let Some((used, pool)) = *self.capacity.lock().unwrap_or_else(|e| e.into_inner()) {
            one(
                out,
                "rexl3_kv_pool_tokens_used",
                "gauge",
                "Tokens reserved by admitted sequences; not allocator usage; refreshed after engine steps.",
                used,
            );
            one(out, "rexl3_kv_pool_tokens", "gauge", "Declared planned KV token capacity; not GPU free memory.", pool);
        }
        if let Some((used, total)) = *self.admission_bytes.lock().unwrap_or_else(|e| e.into_inner()) {
            one(
                out,
                "rexl3_admission_bytes_used",
                "gauge",
                "Admission bytes reserved including pinned cache; not allocator-measured GPU memory.",
                used,
            );
            one(out, "rexl3_admission_bytes_capacity", "gauge", "Engine admission byte capacity.", total);
        }
        // Auto-shrinking families publish their resolved count as a boot fact; others retain the explicit configured count.
        let slots = gb_engine::boot::facts()
            .iter()
            .find(|(k, _)| *k == "slots")
            .and_then(|(_, v)| v.parse::<u64>().ok())
            .unwrap_or_else(|| g(&self.slots));
        if slots > 0 && self.snapshot_at.lock().unwrap_or_else(|e| e.into_inner()).is_some() {
            one(
                out,
                "rexl3_slots_total",
                "gauge",
                "Resolved configured concurrent sequence slots; slots used equals active plus prefilling.",
                slots,
            );
        }
        if let Some(at) = *self.snapshot_at.lock().unwrap_or_else(|e| e.into_inner()) {
            one(
                out,
                "rexl3_engine_snapshot_age_seconds",
                "gauge",
                "Age of last post-step engine snapshot; unchanged while idle or inside a long prefill slice.",
                format!("{:.3}", at.elapsed().as_secs_f64()),
            );
        }
        if let Some(p) = *self.prefill.lock().unwrap_or_else(|e| e.into_inner()) {
            one(out, "rexl3_prefill_tokens_processed_total", "counter", "Prompt tokens reported advanced by successful backend prefill slices, excludes cache restores; includes later failed or cancelled requests.", p.processed);
            one(
                out,
                "rexl3_prefill_tokens_cached_total",
                "counter",
                "Prompt tokens restored from prefix cache, at restore time, not actual backend prefill.",
                p.cached,
            );
        }
        for (h, name, help) in [
            (&self.ttft, "rexl3_request_ttft_seconds", "HTTP middleware entry to first engine token batch observed by decoder; includes body/preprocessing/inbox/queue/prefill, not SSE flush."),
            (&self.submit_ttft, "rexl3_engine_submit_ttft_seconds", "Immediately before engine submit to first engine token batch observed by decoder; includes inbox/queue/prefill, not SSE flush."),
            (&self.batch_gap, "rexl3_token_batch_itl_seconds", "Gaps between consecutive engine token batches observed by decoder; batches not tokens and not SSE chunks."),
            (&self.e2e, "rexl3_request_e2e_seconds", "HTTP middleware entry to generation task terminal result, success or error; excludes client cancellation; not network flush."),
            (&self.preprocessing, "rexl3_request_preprocessing_seconds", "HTTP entry through body parse, template, tokenization and image preparation to engine submit, for submitted requests."),
            (&self.queue, "rexl3_request_queue_seconds", "Engine FIFO enqueue to successful slot admission; excludes inbox/preprocessing and rejected queue timeouts."),
            (&self.prefill_slice, "rexl3_engine_prefill_slice_seconds", "Wall time of engine steps begun with prefilling sequences; can include concurrent decode work, not whole request prefill."),
            (&self.http_headers, "rexl3_http_headers_seconds", "HTTP entry to response header creation, all routes; for streaming excludes terminal body duration."),
        ] { h.render(out, name, help); }
    }
}

pub fn safe_id(value: Option<&str>) -> String {
    if let Some(v) =
        value.filter(|v| !v.is_empty() && v.len() <= 64 && v.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c)))
    {
        return v.to_owned();
    }
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let n = NEXT.fetch_add(1, Relaxed);
    format!("req-{n:016x}")
}
fn route(path: &str) -> &'static str {
    match path {
        "/health" => "/health",
        "/health/live" => "/health/live",
        "/health/ready" => "/health/ready",
        "/v1/health" => "/v1/health",
        "/metrics" => "/metrics",
        "/v1/models" => "/v1/models",
        "/v1/completions" => "/v1/completions",
        "/v1/chat/completions" => "/v1/chat/completions",
        "/v1/chat/completions/render" => "/v1/chat/completions/render",
        "/v1/decisions" => "/v1/decisions",
        _ => "other",
    }
}
pub async fn middleware(State(s): State<Arc<AppState>>, mut req: Request, next: Next) -> Response {
    let at = Instant::now();
    let id = safe_id(req.headers().get("x-request-id").and_then(|v| v.to_str().ok()));
    let path = route(req.uri().path()); // never log unknown user-controlled path or query
    let method = match req.method().as_str() {
        "GET" => "GET",
        "POST" => "POST",
        "HEAD" => "HEAD",
        "OPTIONS" => "OPTIONS",
        _ => "other",
    };
    let m = &s.handle.metrics.observability;
    m.http_entered.fetch_add(1, Relaxed);
    // a generation request arrives until the engine answers its submit (`start`); a request that ends first (refused before it submits,
    // or dropped with its connection) is settled here, when this future ends or is dropped
    let mark = if method == "POST" && matches!(path, "/v1/chat/completions" | "/v1/completions") {
        ArrivalMark::counted(m)
    } else {
        ArrivalMark::default()
    };
    let _settle = Settle(m, mark.clone());
    req.extensions_mut().insert(Arrival(at, mark));
    let mut response = next.run(req).await;
    let status = response.status().as_u16().to_string();
    let i = CODES.iter().position(|c| *c == status).unwrap_or(CODES.len());
    m.http_codes[i].fetch_add(1, Relaxed);
    let ri = ROUTES.iter().position(|r| *r == path).expect("canonical bounded route");
    m.http_route_codes[ri][i].fetch_add(1, Relaxed);
    m.http_headers.record(at.elapsed());
    response.headers_mut().insert("x-request-id", HeaderValue::from_str(&id).expect("sanitized header"));
    eprintln!(
        "rexl3 event=http ts={} method={method} route={path} status={status} headers_ms={:.3} id={id}",
        crate::api::now(),
        at.elapsed().as_secs_f64() * 1000.0
    );
    response
}

/// Own this guard in the SSE body, not the engine producer. Success/error only when the terminal event is actually polled: `[DONE]` for a
/// reply that finished, the error event itself for one that did not (an errored stream closes after it, with no `[DONE]`).
pub struct SseOutcome {
    metrics: Arc<crate::runtime::Metrics>,
    finished: bool,
}
impl SseOutcome {
    pub fn new(metrics: Arc<crate::runtime::Metrics>) -> Self {
        Self { metrics, finished: false }
    }
    pub fn finished(&self) -> bool {
        self.finished
    }
    pub fn terminal(&mut self, error: bool) {
        if !self.finished {
            self.metrics.observability.sse[usize::from(error)].fetch_add(1, Relaxed);
            self.finished = true;
        }
    }
}
impl Drop for SseOutcome {
    fn drop(&mut self) {
        if !self.finished {
            self.metrics.observability.sse[2].fetch_add(1, Relaxed);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loading_does_not_claim_resolved_capacity() {
        let o = Observability::default();
        o.slots.store(8, Relaxed);
        let mut text = String::new();
        o.render(&mut text);
        assert!(!text.contains("rexl3_slots_total"));
        assert!(!text.contains("rexl3_kv_pool_tokens"));
        assert!(!text.contains("rexl3_prefill_tokens_processed_total"));
    }

    #[test]
    fn ids_and_routes_are_bounded_and_content_free() {
        assert_eq!(safe_id(Some("a-Z.1_2")), "a-Z.1_2");
        for v in ["", "a b", "a\nb", "é", &"x".repeat(65)] {
            let id = safe_id(Some(v));
            assert_eq!(id.len(), 20);
            assert!(id.starts_with("req-"));
        }
        assert_eq!(route("/secret-user-text"), "other");
    }
}
