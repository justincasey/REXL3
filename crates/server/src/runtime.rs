//! The engine thread. One OS thread owns the `Engine` (and through it the GPU context); HTTP handlers talk to it over a channel.
//!
//! Requests wait in a FIFO queue **inside the engine thread**: a request is validated and queued when the thread next looks at its
//! inbox (the answer to `EngineHandle::submit` is that verdict: queued, or refused with the reason), admitted strictly in arrival order
//! as slots and memory free up, and its tokens then stream back over the channel it brought. A request can leave the queue four ways:
//! admitted, `--queue-timeout`, the client went away, or the server is shutting down.
//!
//! Failure policy ("never a zombie"): a recoverable step error fails the sequences of that step and is counted, readiness drops after
//! `unready_after` of them in a row, and the engine is declared failed after `exit_after`; a fatal backend error, a panic (also while
//! booting) or too many step errors end the process through `RuntimeConfig::on_fatal` (default: log, give the HTTP side a moment to
//! deliver the errors, exit non-zero), and every request in flight or queued is answered with an error, never with a normal finish.

use crate::flow::Sender as StreamSender;
use gb_engine::score::{ScoreRequest, Scored};
use gb_engine::{Backend, Engine, FinishReason, Request, ServeEngine, SubmitError};
pub use gb_model::exit::FatalExit;
use gb_sampling::Sampling;
use gb_sched::AdmitError;
use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

#[derive(Debug, Clone)]
pub struct GenRequest {
    pub prompt: Vec<u32>,
    pub sampling: Sampling,
    pub max_new: usize,
    pub stop: Vec<u32>,
    pub draft: bool,
    /// Images of the prompt (placeholders already expanded in `prompt`); empty for text.
    pub images: Vec<gb_engine::ImageSpan>,
    /// May use the prefix cache (`"prefix_cache": false` in the request opts out).
    pub cache: bool,
    /// The reply's reasoning budget (`gb_engine::ReasoningBudget`): `None` for every request that did not ask for one.
    pub reasoning_budget: Option<gb_engine::ReasoningBudget>,
}

/// What a running sequence sends back.
#[derive(Debug, Clone, PartialEq)]
pub enum SeqMsg {
    Tokens(Vec<u32>),
    Done(DoneInfo),
}

impl crate::flow::Message for SeqMsg {
    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + match self {
                Self::Tokens(v) => v.capacity().saturating_mul(4),
                Self::Done(d) => d.error.as_ref().map_or(0, |e| e.message.capacity()),
            }
    }
    fn failure(e: SeqError) -> Self {
        Self::Done(DoneInfo::failed(e))
    }
}

/// Why a request that was accepted did not finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailKind {
    /// The engine failed this sequence (a step error, a row that could not be sampled, a prefill error).
    Engine,
    /// It waited in the queue longer than `--queue-timeout`.
    QueueTimeout,
    /// The server is shutting down: it was still queued, or its reply was cut off at the end of the grace period.
    Shutdown,
    /// The engine stopped (a fatal error or a panic) or went away.
    EngineDown,
    RequestTimeout,
    Backpressure,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SeqError {
    pub kind: FailKind,
    pub message: String,
}

impl SeqError {
    fn new(kind: FailKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoneInfo {
    pub finish: Option<FinishReason>,
    /// Set when the sequence did not finish: the reply is cut short and must be reported as an error.
    pub error: Option<SeqError>,
    pub rounds: u64,
    pub drafted: u64,
    pub accepted: u64,
    /// Prompt tokens a prefix-cache hit supplied.
    pub cached_tokens: usize,
}

impl DoneInfo {
    pub fn failed(e: SeqError) -> Self {
        Self { finish: None, error: Some(e), rounds: 0, drafted: 0, accepted: 0, cached_tokens: 0 }
    }
}

/// Why `EngineHandle::submit` did not queue a request.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// The queue is at `--queue-depth`: try again later.
    QueueFull {
        depth: usize,
    },
    /// The engine is still loading.
    NotReady,
    /// The server is shutting down.
    ShuttingDown,
    /// The engine stopped, or is no longer running.
    EngineDown(String),
    RequestTimeout,
    /// Even a lower bound on the request's prefill (`ServeEngine::prefill_floor`, from the longest cached prefix) exceeds what is left of its
    /// deadline (`--request-timeout`): it could never finish, so it is refused before it holds a slot or a page.
    DeadlineTooShort {
        floor: Duration,
        left: Duration,
    },
    /// The engine refused the request itself: malformed, or it can never fit (see `Engine::validate`).
    Rejected(SubmitError),
}

/// A queued request's handle for `EngineHandle::cancel`; valid from the moment `submit` returns it.
pub type Ticket = u64;

enum Cmd {
    Score {
        ticket: Ticket,
        req: ScoreRequest,
        out: oneshot::Sender<Result<Scored, SeqError>>,
        ack: oneshot::Sender<Result<(), Refusal>>,
        deadline: Instant,
    },
    Submit {
        ticket: Ticket,
        deadline: Instant,
        req: GenRequest,
        events: StreamSender<SeqMsg>,
        ack: oneshot::Sender<Result<(), Refusal>>,
    },
    Cancel(Ticket),
    /// Stop admitting, finish what is running until `deadline`, then end the rest as errors and stop.
    Drain {
        deadline: Instant,
    },
    /// Stop now.
    Shutdown,
}

/// What the server reports as its state: `/health` is 200 only in `Ready`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Loading = 0,
    Ready = 1,
    /// Shutting down: not admitting, finishing what runs.
    Draining = 2,
    /// Consecutive recoverable step errors: still serving, but not to be sent traffic.
    Degraded = 3,
    /// The engine stopped for good.
    Failed = 4,
}

impl State {
    pub fn name(self) -> &'static str {
        match self {
            State::Loading => "loading",
            State::Ready => "ok",
            State::Draining => "draining",
            State::Degraded => "degraded",
            State::Failed => "failed",
        }
    }
}

#[derive(Default)]
pub struct Metrics {
    pub decisions_admitted: AtomicU64,
    pub decisions_completed: AtomicU64,
    pub decisions_cancelled: AtomicU64,
    pub decision_prompt_tokens: AtomicU64,
    pub decisions_cached_tokens: AtomicU64,
    score_supported: std::sync::atomic::AtomicBool,
    pub observability: crate::observability::Observability,
    /// Requests admitted to the engine (they got a slot).
    pub requests: AtomicU64,
    /// Requests that never ran: queue full, can never fit, malformed, server shutting down, queue timeout. Once per request.
    pub rejected: AtomicU64,
    pub queue_full: AtomicU64,
    pub queue_timeouts: AtomicU64,
    pub request_timeouts: AtomicU64,
    /// Requests refused at once because even a lower bound on their prefill exceeded their deadline (`Refusal::DeadlineTooShort`).
    pub deadline_refusals: AtomicU64,
    pub stream_failures: AtomicU64,
    progress: Mutex<Option<(Instant, Duration, &'static str)>>,
    /// Engine cancellations before normal completion (client departures or server stop-string cuts).
    pub cancelled: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub completion_tokens: AtomicU64,
    pub drafted: AtomicU64,
    pub accepted: AtomicU64,
    pub rounds: AtomicU64,
    /// Sequences decoding, sequences still prefilling, requests waiting for a slot.
    pub active: AtomicU64,
    pub prefilling: AtomicU64,
    pub queued: AtomicU64,
    /// Engine rounds that returned an error (recoverable or fatal).
    pub step_errors: AtomicU64,
    /// Sequences the engine failed (one bad row, a prefill or commit error, a failed step).
    pub failed_sequences: AtomicU64,
    state: AtomicU8,
    /// Submits sent and not yet looked at by the engine thread: bounds the inbox while a long prefill slice runs.
    inbox: AtomicU64,
    /// The most tokens (prompt plus generation) one request may hold, as the engine reported when it was built (`ServeEngine::request_limit`);
    /// 0 until then and for an engine that declares none.
    request_limit: AtomicU64,
    /// The engine's speculation counters, refreshed after every step (per-position acceptance, round phase times).
    pub spec: std::sync::Mutex<gb_spec::stats::SpecStats>,
    /// The prefix cache's counters and gauges, refreshed after every step.
    pub prefix: std::sync::Mutex<gb_engine::PrefixStats>,
    /// The backend's CUDA-graph cache counters, refreshed after every step; `None` for an engine without graphs (nothing is exported then).
    pub graphs: std::sync::Mutex<Option<gb_engine::GraphCounters>>,
    /// The whole boot, including a component's warm preflight. Defaults to complete; a component
    /// that must finish before readiness holds it false, and the runtime sets it once `make` returns.
    ready_complete: std::sync::atomic::AtomicBool,
}

impl Metrics {
    pub fn state(&self) -> State {
        match self.state.load(Ordering::Acquire) {
            0 => State::Loading,
            1 => State::Ready,
            2 => State::Draining,
            3 => State::Degraded,
            _ => State::Failed,
        }
    }
    fn set_state(&self, s: State) {
        self.state.store(s as u8, Ordering::Release);
    }
    /// The engine's per-request limit in tokens (the smaller of its window and its KV pool), once it is built; `None` before that, or when it
    /// declares none. Published before the server turns ready, so a request that is accepted always sees it.
    pub fn request_limit(&self) -> Option<u64> {
        Some(self.request_limit.load(Ordering::Acquire)).filter(|&n| n > 0)
    }
    /// The limit a client should plan against: the engine's, never above the configured `window`; the window until the engine has said.
    pub fn effective_limit(&self, window: usize) -> usize {
        self.request_limit().map_or(window, |n| (n as usize).min(window))
    }
    fn set_request_limit(&self, limit: Option<u64>) {
        self.request_limit.store(limit.unwrap_or(0), Ordering::Release);
    }
    /// Move from `from` to `to` unless something else (a drain, a failure) has changed the state meanwhile.
    fn transition(&self, from: State, to: State) -> bool {
        self.state.compare_exchange(from as u8, to as u8, Ordering::AcqRel, Ordering::Acquire).is_ok()
    }
    /// The engine is up and healthy: what `/health` and `/health/ready` report.
    pub fn ready(&self) -> bool {
        self.state() == State::Ready && !self.wedged() && self.ready_complete()
    }
    /// Whether the whole boot is complete, including the warm preflight. A serving component
    /// (e.g. Qwen n-gram warm) may keep this false until it has finished, so readiness and
    /// admission stay closed. Default: complete (set by the runtime once `make` returns).
    pub fn set_ready_complete(&self, complete: bool) {
        self.ready_complete.store(complete, Ordering::Release);
    }
    pub fn ready_complete(&self) -> bool {
        self.ready_complete.load(Ordering::Acquire)
    }
    fn heartbeat(&self, phase: &'static str, budget: Duration) {
        *self.progress.lock().unwrap_or_else(|e| e.into_inner()) = Some((Instant::now(), budget, phase));
    }
    /// Enter a busy phase without inventing progress on every no-op engine iteration.
    fn begin_work(&self, phase: &'static str, budget: Duration) {
        let mut progress = self.progress.lock().unwrap_or_else(|e| e.into_inner());
        match progress.as_mut() {
            Some((at, old_budget, old_phase)) if *old_phase != "idle" => {
                let _ = at;
                *old_budget = budget;
                *old_phase = phase;
            }
            _ => *progress = Some((Instant::now(), budget, phase)),
        }
    }
    pub fn heartbeat_age(&self) -> Duration {
        self.progress.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(Duration::ZERO, |(t, _, _)| t.elapsed())
    }
    pub fn wedged(&self) -> bool {
        self.progress.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|(t, b, p)| *p != "idle" && t.elapsed() > *b)
    }
}

/// Most submits that may wait unseen in the inbox (the engine thread cannot answer while a prefill slice runs).
// Bound unseen retained prompt vectors independently of raw-body credits.
// At the advertised 262K context, 128 pending vectors occupy at most 128 MiB.
// Image features retain their separate allocation-owned weighted leases.
const INBOX_CAP: u64 = 128;
/// How often an idle engine's gather window looks again at the requests still arriving (`run`).
const GATHER_POLL: Duration = Duration::from_millis(1);

/// Cloneable front for the engine thread.
#[derive(Clone)]
pub struct EngineHandle {
    tx: Sender<Cmd>,
    pub metrics: Arc<Metrics>,
    tickets: Arc<AtomicU64>,
    queue_depth: usize,
    pub request_timeout: Duration,
    pub write_stall: Duration,
    boot_cancel: Arc<AtomicBool>,
    decision_timeout: Duration,
}

impl EngineHandle {
    /// The most submits that may wait unseen in the inbox: [`INBOX_CAP`], or `--queue-depth` when the operator raised it past that (a wide
    /// decode window seats more than 128 streams, and a burst of that many requests arrives while the engine thread is inside one step).
    fn inbox_cap(&self) -> u64 {
        INBOX_CAP.max(self.queue_depth as u64)
    }

    pub fn decision_timeout(&self) -> Duration {
        self.decision_timeout
    }
    pub fn score_supported(&self) -> bool {
        self.metrics.score_supported.load(Ordering::Acquire)
    }
    /// Guard exists BEFORE enqueue acknowledgement: aborting either await withdraws the score job.
    pub async fn score(&self, req: ScoreRequest) -> Result<(CancelGuard, oneshot::Receiver<Result<Scored, SeqError>>), Refusal> {
        self.score_with_timeout(req, self.decision_timeout).await
    }
    pub async fn score_with_timeout(
        &self,
        req: ScoreRequest,
        timeout: Duration,
    ) -> Result<(CancelGuard, oneshot::Receiver<Result<Scored, SeqError>>), Refusal> {
        match self.metrics.state() {
            State::Loading => return Err(Refusal::NotReady),
            State::Draining => return Err(Refusal::ShuttingDown),
            State::Failed => return Err(Refusal::EngineDown("the engine has stopped".into())),
            _ => {}
        }
        if !self.metrics.ready_complete() {
            return Err(Refusal::NotReady);
        }
        if !self.score_supported() {
            return Err(Refusal::Rejected(gb_engine::score::unsupported()));
        }
        if self.metrics.inbox.fetch_add(1, Ordering::AcqRel) >= self.inbox_cap() {
            self.metrics.inbox.fetch_sub(1, Ordering::AcqRel);
            self.metrics.rejected.fetch_add(1, Ordering::Relaxed);
            self.metrics.queue_full.fetch_add(1, Ordering::Relaxed);
            return Err(Refusal::QueueFull { depth: self.queue_depth });
        }
        let ticket = self.tickets.fetch_add(1, Ordering::Relaxed) + 1;
        let guard = CancelGuard::new(self.clone(), ticket);
        let (ack, verdict) = oneshot::channel();
        let (out, row) = oneshot::channel();
        let deadline = Instant::now().checked_add(timeout).unwrap_or_else(Instant::now);
        if self.tx.send(Cmd::Score { ticket, req, out, ack, deadline }).is_err() {
            self.metrics.inbox.fetch_sub(1, Ordering::AcqRel);
            return Err(Refusal::EngineDown("the engine has stopped".into()));
        }
        match tokio::time::timeout_at(deadline.into(), verdict).await {
            Ok(Ok(Ok(()))) => Ok((guard, row)),
            Ok(Ok(Err(e))) => Err(e),
            Ok(Err(_)) => Err(Refusal::EngineDown("the engine stopped before score acknowledgement".into())),
            Err(_) => Err(Refusal::RequestTimeout),
        }
    }
    /// Queue a request. Resolves when the engine thread has looked at it: `Ok` means it is queued (it runs when a slot and memory are
    /// free, in arrival order; tokens, errors and the end arrive on `events`), `Err` says why it was refused. A streaming client can
    /// be given its response headers as soon as this resolves.
    pub async fn submit(&self, req: GenRequest, events: StreamSender<SeqMsg>) -> Result<Ticket, Refusal> {
        self.submit_until(req, events, Instant::now() + self.request_timeout).await
    }

    pub async fn submit_until(
        &self,
        req: GenRequest,
        events: StreamSender<SeqMsg>,
        deadline: Instant,
    ) -> Result<Ticket, Refusal> {
        match self.metrics.state() {
            State::Loading => return Err(Refusal::NotReady),
            State::Draining => return Err(Refusal::ShuttingDown),
            State::Failed => return Err(Refusal::EngineDown("the engine has stopped".into())),
            State::Ready | State::Degraded => {}
        }
        if !self.metrics.ready_complete() {
            return Err(Refusal::NotReady);
        }
        if self.metrics.inbox.fetch_add(1, Ordering::AcqRel) >= self.inbox_cap() {
            self.metrics.inbox.fetch_sub(1, Ordering::AcqRel);
            self.metrics.rejected.fetch_add(1, Ordering::Relaxed);
            self.metrics.queue_full.fetch_add(1, Ordering::Relaxed);
            return Err(Refusal::QueueFull { depth: self.queue_depth });
        }
        let ticket = self.tickets.fetch_add(1, Ordering::Relaxed) + 1;
        let (ack, verdict) = oneshot::channel();
        let mut guard = CancelGuard::new(self.clone(), ticket);
        if self.tx.send(Cmd::Submit { ticket, deadline, req, events, ack }).is_err() {
            self.metrics.inbox.fetch_sub(1, Ordering::AcqRel);
            return Err(Refusal::EngineDown("the engine has stopped".into()));
        }
        match tokio::time::timeout_at(deadline.into(), verdict).await {
            Ok(Ok(Ok(()))) => {
                guard.armed = false;
                Ok(ticket)
            }
            Ok(Ok(Err(r))) => Err(r),
            Ok(Err(_)) => Err(Refusal::EngineDown("the engine stopped before it looked at the request".into())),
            Err(_) => Err(Refusal::RequestTimeout),
        }
    }

    /// Withdraw a request: out of the queue, or (admitted, prefilling or decoding) stopped with its slot and KV released. A ticket
    /// that is finished or unknown is ignored.
    pub fn cancel(&self, ticket: Ticket) {
        let _ = self.tx.send(Cmd::Cancel(ticket));
    }

    /// Begin a graceful stop: no new requests, queued ones get a 503, running ones finish for up to `grace`, after which what is left
    /// is ended as errors and the engine thread returns.
    pub fn drain(&self, grace: Duration) {
        self.boot_cancel.store(true, Ordering::Release);
        // readiness drops and new requests are refused at once, not when the engine thread next looks at its inbox (a prefill slice
        // can take seconds)
        for from in [State::Ready, State::Degraded, State::Loading] {
            if self.metrics.transition(from, State::Draining) {
                break;
            }
        }
        let _ = self.tx.send(Cmd::Drain { deadline: Instant::now() + grace });
    }

    /// Stop the engine now.
    pub fn shutdown(&self) {
        self.boot_cancel.store(true, Ordering::Release);
        let _ = self.tx.send(Cmd::Shutdown);
    }
}

/// Ends a request when dropped: hold one for as long as someone is waiting for the reply, so a client that disconnects while its
/// request is queued or prefilling gives its slot and KV back within one round (nothing else notices before the first token).
pub struct CancelGuard {
    handle: EngineHandle,
    ticket: Ticket,
    armed: bool,
}

impl CancelGuard {
    pub fn new(handle: EngineHandle, ticket: Ticket) -> Self {
        Self { handle, ticket, armed: true }
    }
    pub fn ticket(&self) -> Ticket {
        self.ticket
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if self.armed {
            self.handle.cancel(self.ticket);
        }
    }
}

/// Why the engine cannot go on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FatalInfo {
    pub reason: String,
    /// The engine thread panicked (as opposed to a reported fatal error).
    pub panicked: bool,
    /// How the process ends (`RuntimeConfig::fatal_exit`): a tensor-parallel group's ends with `_exit`, not `exit`.
    pub exit: FatalExit,
}

impl FatalInfo {
    /// The process exit status for this failure: 101 for a panic (Rust's own), 70 (EX_SOFTWARE) otherwise.
    pub fn exit_code(&self) -> i32 {
        if self.panicked {
            101
        } else {
            70
        }
    }
}

#[derive(Clone)]
pub struct RuntimeConfig {
    /// Whole decisions request deadline (HTTP preprocessing + all question prefills).
    pub decision_timeout: Duration,
    /// Requests that may wait for a slot; one more is refused (429).
    pub queue_depth: usize,
    /// How long a request may wait for a slot before it is answered 503.
    pub queue_timeout: Duration,
    pub request_timeout: Duration,
    pub write_stall: Duration,
    pub heartbeat_stale: Duration,
    pub loading_budget: Duration,
    pub prefill_budget: Duration,
    /// Consecutive recoverable step errors after which readiness drops until a step succeeds.
    pub unready_after: u32,
    /// Consecutive recoverable step errors after which the engine is declared failed (a not-ready server that sees no traffic would
    /// otherwise never recover): the process exits for its supervisor to restart.
    pub exit_after: u32,
    /// Called once, on the engine thread, after every request in flight has been answered with an error, when the engine cannot go on.
    /// The default exits the process with `FatalInfo::exit_code` after giving the HTTP side a moment to deliver the errors.
    pub on_fatal: Arc<dyn Fn(&FatalInfo) + Send + Sync>,
    /// On a fatal engine exit, leak the engine instead of dropping it. Set for a tensor-parallel group: a destructor that frees device
    /// memory can block behind a collective that a lost peer will never join, and `die` (which fails the requests by name and ends the
    /// process) would then never run. Off elsewhere: every other family drops its engine as before.
    pub leak_engine_on_fatal: bool,
    /// How the default `on_fatal` ends the process. `FatalExit::Immediate` (a tensor-parallel group, `gb_model::exit::fatal_exit_mode`):
    /// stderr flushed and `_exit`, with no atexit handler, so the CUDA runtime's exit-time teardown of a context whose communicator was
    /// aborted and whose engine was leaked cannot fault (observed: status 139 where 70 was logged). Every other family keeps `exit`.
    pub fatal_exit: FatalExit,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            decision_timeout: Duration::from_secs(1800),
            queue_depth: 64,
            queue_timeout: Duration::from_secs(300),
            request_timeout: Duration::from_secs(1800),
            write_stall: Duration::from_secs(30),
            heartbeat_stale: Duration::from_secs(60),
            loading_budget: Duration::from_secs(1800),
            prefill_budget: Duration::from_secs(1800),
            unready_after: 3,
            exit_after: 10,
            leak_engine_on_fatal: false,
            fatal_exit: FatalExit::Exit,
            on_fatal: Arc::new(|info| {
                eprintln!("rexl3 level=fatal event=exit code={} reason=\"{}\"", info.exit_code(), info.reason);
                if info.exit == FatalExit::Exit {
                    // `exit` runs the CUDA runtime's atexit teardown, which a faulted context may never finish: do not stay a zombie then
                    std::thread::spawn(|| {
                        std::thread::sleep(Duration::from_secs(5));
                        std::process::abort()
                    });
                }
                gb_model::exit::end_process(info.exit_code(), info.exit)
            }),
        }
    }
}

/// Start the engine thread. `make` runs on that thread, so a CUDA context is created where it is used.
pub fn spawn<B, F>(make: F) -> (EngineHandle, JoinHandle<()>)
where
    B: Backend + 'static,
    F: FnOnce() -> Engine<B> + Send + 'static,
{
    spawn_cfg(make, RuntimeConfig::default())
}

/// [`spawn`] with a configuration. The one place a concrete `Engine<B>` is erased on the server's side: the loop (`run`) is written
/// once, over `dyn ServeEngine`, and serves this, [`spawn_dyn`] and [`spawn_with`] alike.
pub fn spawn_cfg<B, F>(make: F, cfg: RuntimeConfig) -> (EngineHandle, JoinHandle<()>)
where
    B: Backend + 'static,
    F: FnOnce() -> Engine<B> + Send + 'static,
{
    spawn_with(Box::new(move || Box::new(make()) as Box<dyn ServeEngine>), cfg)
}

/// [`spawn`] for an engine whose backend type is erased: a model family's `build` returns `Box<dyn ServeEngine>`, so the server
/// does not name any family. `make` runs on the engine thread.
pub fn spawn_dyn(make: Box<dyn FnOnce() -> Box<dyn ServeEngine> + Send>) -> (EngineHandle, JoinHandle<()>) {
    spawn_with(make, RuntimeConfig::default())
}

/// [`spawn_dyn`] with a configuration.
pub fn spawn_with(make: Box<dyn FnOnce() -> Box<dyn ServeEngine> + Send>, cfg: RuntimeConfig) -> (EngineHandle, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel();
    let metrics = Arc::new(Metrics::default());
    let m = metrics.clone();
    let queue_depth = cfg.queue_depth;
    let request_timeout = cfg.request_timeout;
    let write_stall = cfg.write_stall;
    let boot_cancel = Arc::new(AtomicBool::new(false));
    let abort_boot = boot_cancel.clone();
    m.heartbeat("loading", cfg.loading_budget);
    let decision_timeout = cfg.decision_timeout;
    let th = std::thread::Builder::new()
        .name("rexl3-engine".into())
        .spawn(move || {
            // The queue and the streams live outside the unwind boundary: after a panic they are still here to be answered.
            let mut lp = Loop::default();
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                gb_engine::boot::install_cancel(abort_boot);
                gb_engine::boot::checkpoint();
                let mut engine = make();
                gb_engine::boot::checkpoint();
                m.set_ready_complete(true);
                m.score_supported.store(engine.score_supported(), Ordering::Release);
                snapshot(engine.as_ref(), &m);
                m.set_request_limit(engine.request_limit()); // before the server turns ready: no accepted request sees the window alone
                gb_engine::boot::mark("ready", None);
                m.heartbeat("idle", cfg.heartbeat_stale);
                m.transition(State::Loading, State::Ready); // not if a drain began while the weights loaded
                let mut exit = run(engine.as_mut(), &rx, &mut lp, &m, &cfg);
                if matches!(exit, Exit::Stopped) {
                    if let Err(e) = engine.quiesce() {
                        // A fatal exit like any other, and it must reach the leak below: this used to `return` from here, which dropped
                        // the engine (its device buffers, its graphs, the communicator) while the group's abort thread was still inside
                        // the communicator, the one fatal path of a lost group that freed the device state before the process ended.
                        exit = Exit::Fatal(format!("engine quiescence failed: {e}"));
                    }
                }
                if cfg.leak_engine_on_fatal && matches!(exit, Exit::Fatal(_)) {
                    // The engine is not dropped on the way out: `die` below fails every request by name and ends the process, and a
                    // destructor that frees device memory can block on a device that is stuck (a tensor-parallel rank in a collective
                    // with a lost peer: the clients would see no error for the whole progress-watchdog budget, and `die` never ran).
                    std::mem::forget(engine);
                }
                exit
            }));
            let info = match outcome {
                Ok(Exit::Stopped) => return,
                Ok(Exit::Fatal(reason)) => FatalInfo { reason, panicked: false, exit: cfg.fatal_exit },
                Err(payload) if payload.is::<gb_engine::boot::Cancelled>() => {
                    eprintln!("rexl3 level=info event=boot_cancelled quiesced=true");
                    m.set_state(State::Draining);
                    return;
                }
                Err(payload) => FatalInfo { reason: panic_message(payload.as_ref()), panicked: true, exit: cfg.fatal_exit },
            };
            die(&info, &rx, &mut lp, &m, &cfg);
        })
        .expect("spawn the engine thread");
    (
        EngineHandle {
            tx,
            metrics,
            tickets: Arc::new(AtomicU64::new(0)),
            queue_depth,
            request_timeout,
            write_stall,
            boot_cancel,
            decision_timeout,
        },
        th,
    )
}

#[cfg(test)]
mod last_progress_tests {
    use super::*;
    #[test]
    fn empty_busy_iterations_do_not_invent_progress_or_idle() {
        let m = Metrics::default();
        m.set_state(State::Ready);
        m.heartbeat("idle", Duration::from_millis(10));
        m.begin_work("decode", Duration::from_millis(10));
        std::thread::sleep(Duration::from_millis(25));
        for _ in 0..100 {
            m.begin_work("decode", Duration::from_millis(10));
        }
        assert!(m.wedged());
        assert!(!m.ready());
        m.heartbeat("decode", Duration::from_millis(10)); // actual successful engine progress
        assert!(!m.wedged());
    }
    #[test]
    fn legitimate_prefill_and_true_idle_have_distinct_allowances() {
        let m = Metrics::default();
        m.heartbeat("idle", Duration::from_millis(5));
        m.begin_work("prefill", Duration::from_millis(200));
        std::thread::sleep(Duration::from_millis(20));
        m.begin_work("prefill", Duration::from_millis(200));
        assert!(!m.wedged());
        assert!(m.heartbeat_age() >= Duration::from_millis(20));
        m.heartbeat("idle", Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        assert!(!m.wedged(), "sleeping with no engine work is legitimate idle");
    }
}

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned());
    format!("the engine thread panicked: {}", msg.unwrap_or_else(|| "(no message)".into()))
}

type QueuedScore = (ScoreRequest, oneshot::Sender<Result<Scored, SeqError>>, Instant);
struct Queued {
    score: Option<QueuedScore>,
    ticket: Ticket,
    req: Request,
    /// The request's reasoning budget, handed to the engine with it.
    budget: Option<gb_engine::ReasoningBudget>,
    events: StreamSender<SeqMsg>,
    since: Instant,
    deadline: Instant,
}

impl Queued {
    fn closed(&self) -> bool {
        self.score.as_ref().map_or_else(|| self.events.is_closed(), |s| s.1.is_closed())
    }
    fn fail(self, e: SeqError) {
        if let Some((_, out, _)) = self.score {
            let _ = out.send(Err(e));
        } else {
            let _ = self.events.send(SeqMsg::Done(DoneInfo::failed(e)));
        }
    }
}
struct ScoreStream {
    ticket: Ticket,
    out: oneshot::Sender<Result<Scored, SeqError>>,
    deadline: Instant,
}
struct Stream {
    ticket: Ticket,
    tx: StreamSender<SeqMsg>,
    deadline: Instant,
}

#[derive(Default)]
struct Loop {
    queue: VecDeque<Queued>,
    /// Admitted sequences by engine id.
    streams: HashMap<u64, Stream>,
    scores: HashMap<u64, ScoreStream>,
    /// Ticket to engine id, for the admitted ones.
    admitted: HashMap<Ticket, u64>,
    draining: Option<Instant>,
    /// Recoverable step errors in a row.
    consecutive: u32,
}

enum Exit {
    /// Shut down on request, or drained: not a failure.
    Stopped,
    Fatal(String),
}

fn run(engine: &mut dyn ServeEngine, rx: &Receiver<Cmd>, lp: &mut Loop, m: &Metrics, cfg: &RuntimeConfig) -> Exit {
    loop {
        // Idle: sleep until a command arrives or the next deadline (a queue timeout, the end of the drain) falls due. Busy: take
        // whatever is waiting and keep stepping.
        let mut cmds = Vec::new();
        if !engine.busy() {
            let wake = next_deadline(lp, cfg);
            let got = match wake {
                Some(at) => rx.recv_timeout(at.saturating_duration_since(Instant::now())),
                None => rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
            };
            match got {
                Ok(c) => {
                    // a request that wakes the idle engine waits up to the engine's gather window for the requests sent with it (their
                    // prompts are then filled together), but only while some are still arriving: in the HTTP layer and not yet in this
                    // inbox (`Observability::arriving` counts the submits here too, unanswered until the commands below run). A request
                    // that arrives alone starts at once.
                    let gather = if matches!(c, Cmd::Submit { .. }) { engine.gather_window() } else { Duration::ZERO };
                    cmds.push(c);
                    let until = Instant::now() + gather;
                    let submits = |cmds: &[Cmd]| cmds.iter().filter(|c| matches!(c, Cmd::Submit { .. })).count() as u64;
                    while !gather.is_zero()
                        && Instant::now() < until
                        && m.observability.arriving.load(std::sync::atomic::Ordering::Relaxed) > submits(&cmds)
                    {
                        // short slices: a request that arrived and was refused before it submitted stops the wait too
                        let slice = until.saturating_duration_since(Instant::now()).min(GATHER_POLL);
                        match rx.recv_timeout(slice) {
                            Ok(c) => cmds.push(c),
                            Err(RecvTimeoutError::Timeout) => {}
                            Err(RecvTimeoutError::Disconnected) => break,
                        }
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Exit::Stopped, // every sender is gone
            }
        }
        while let Ok(c) = rx.try_recv() {
            cmds.push(c);
        }
        for c in cmds {
            match c {
                Cmd::Shutdown => {
                    engine.abort_all();
                    fail_everything(lp, SeqError::new(FailKind::Shutdown, "the server is shutting down"));
                    return Exit::Stopped;
                }
                Cmd::Cancel(t) => cancel(engine, lp, m, t),
                Cmd::Drain { deadline } => start_drain(lp, m, deadline),
                Cmd::Submit { ticket, deadline, req, events, ack } => {
                    enqueue(engine, lp, m, cfg, ticket, deadline, req, events, ack)
                }
                Cmd::Score { ticket, req, out, ack, deadline } => {
                    enqueue_score(engine, lp, m, cfg, ticket, req, out, ack, deadline)
                }
            }
        }
        reap_closed(engine, lp, m);
        expire_scores(engine, lp, m);
        expire_requests(engine, lp, m);
        expire_queue(lp, m, cfg);
        admit(engine, lp, m);
        let request_deadline = lp.streams.values().map(|s| s.deadline).chain(lp.scores.values().map(|s| s.deadline)).min();
        let work_deadline = lp.draining.into_iter().chain(request_deadline).min();
        engine.set_slice_deadline(work_deadline);
        if lp.draining.is_some() && !engine.supports_slice_deadline() {
            engine.abort_all();
            fail_everything(lp, SeqError::new(FailKind::Shutdown, "this engine cannot bound draining work"));
            return Exit::Stopped;
        }
        if lp.draining.is_some_and(|d| Instant::now() >= d) {
            engine.abort_all();
            fail_everything(lp, SeqError::new(FailKind::Shutdown, "the server shut down before this reply was complete"));
            return Exit::Stopped;
        }
        if engine.busy() {
            // only a round that decoded proves the device is sound: a step that merely advanced a prefill does not clear the count
            let decoding = engine.active() > 0;
            let phase = if engine.prefilling() > 0 { "prefill" } else { "decode" };
            let budget = if engine.prefilling() > 0 { cfg.prefill_budget } else { cfg.heartbeat_stale };
            m.begin_work(phase, budget);
            let prefill = engine.prefilling() > 0;
            let step_at = Instant::now();
            let result = engine.step();
            if prefill {
                m.observability.prefill_slice.record(step_at.elapsed());
            }
            match result {
                Ok(progress) => {
                    if progress {
                        m.heartbeat(phase, budget);
                    }
                    if !progress && engine.busy() {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    if decoding {
                        lp.consecutive = 0;
                        if m.transition(State::Degraded, State::Ready) {
                            eprintln!("rexl3 level=info event=recovered");
                        }
                    }
                }
                Err(e) if e.is_fatal() => {
                    m.step_errors.fetch_add(1, Ordering::Relaxed);
                    return Exit::Fatal(e.to_string());
                }
                Err(e) => {
                    m.step_errors.fetch_add(1, Ordering::Relaxed);
                    lp.consecutive += 1;
                    let n = engine.fail_active(&e.to_string());
                    eprintln!(
                        "rexl3 level=error event=step_failed failed_sequences={n} consecutive={} error=\"{e}\"",
                        lp.consecutive
                    );
                    if lp.consecutive >= cfg.exit_after {
                        return Exit::Fatal(format!("{} consecutive step errors, the last: {e}", lp.consecutive));
                    }
                    if lp.consecutive >= cfg.unready_after && m.transition(State::Ready, State::Degraded) {
                        eprintln!("rexl3 level=error event=degraded consecutive={}", lp.consecutive);
                    }
                }
            }
        }
        expire_scores(engine, lp, m);
        expire_requests(engine, lp, m);
        route(engine, lp, m);
        if !engine.busy() {
            m.heartbeat("idle", cfg.heartbeat_stale);
        }
        snapshot(engine, m);
        m.active.store(engine.active() as u64, Ordering::Relaxed);
        m.prefilling.store(engine.prefilling() as u64, Ordering::Relaxed);
        m.queued.store(lp.queue.len() as u64, Ordering::Relaxed);
        if let Ok(mut g) = m.spec.lock() {
            *g = engine.spec_stats();
        }
        if let Some(deadline) = lp.draining {
            if !engine.busy() && lp.queue.is_empty() {
                return Exit::Stopped;
            }
            if Instant::now() >= deadline {
                let n = engine.abort_all();
                eprintln!("rexl3 level=warn event=drain_deadline truncated={n}");
                fail_everything(lp, SeqError::new(FailKind::Shutdown, "the server shut down before this reply was complete"));
                return Exit::Stopped;
            }
        }
    }
}

fn snapshot(engine: &dyn ServeEngine, m: &Metrics) {
    let o = &m.observability;
    *o.capacity.lock().unwrap_or_else(|e| e.into_inner()) = engine.kv_stats();
    *o.admission_bytes.lock().unwrap_or_else(|e| e.into_inner()) = engine.admission_stats();
    *o.prefill.lock().unwrap_or_else(|e| e.into_inner()) = engine.prefill_stats();
    *m.prefix.lock().unwrap_or_else(|e| e.into_inner()) = engine.prefix_stats();
    *m.graphs.lock().unwrap_or_else(|e| e.into_inner()) = engine.graph_counters();
    *o.snapshot_at.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
}

/// The next moment the loop must wake while idle.
fn next_deadline(lp: &Loop, cfg: &RuntimeConfig) -> Option<Instant> {
    let queue = lp.queue.front().map(|q| q.since + cfg.queue_timeout);
    queue
        .into_iter()
        .chain(lp.draining)
        .chain(lp.queue.iter().map(|q| q.deadline))
        .chain(lp.streams.values().map(|s| s.deadline))
        .chain(lp.scores.values().map(|s| s.deadline))
        .min()
}

#[allow(clippy::too_many_arguments)]
fn enqueue_score(
    engine: &dyn ServeEngine,
    lp: &mut Loop,
    m: &Metrics,
    cfg: &RuntimeConfig,
    ticket: Ticket,
    req: ScoreRequest,
    out: oneshot::Sender<Result<Scored, SeqError>>,
    ack: oneshot::Sender<Result<(), Refusal>>,
    deadline: Instant,
) {
    m.inbox.fetch_sub(1, Ordering::AcqRel);
    let refusal = if lp.draining.is_some() {
        Some(Refusal::ShuttingDown)
    } else if let Err(e) = engine.score_validate(&req) {
        Some(Refusal::Rejected(e))
    } else if lp.queue.len() >= cfg.queue_depth {
        m.queue_full.fetch_add(1, Ordering::Relaxed);
        Some(Refusal::QueueFull { depth: cfg.queue_depth })
    } else {
        None
    };
    if let Some(e) = refusal {
        m.rejected.fetch_add(1, Ordering::Relaxed);
        let _ = ack.send(Err(e));
        return;
    }
    if ack.is_closed() || out.is_closed() {
        m.decisions_cancelled.fetch_add(1, Ordering::Relaxed);
        return;
    }
    // The ordinary FIFO holds BOTH job kinds; this inert container is never validated/submitted as generation.
    let placeholder = Request {
        prompt: vec![],
        sampling: Default::default(),
        max_new: 0,
        stop: vec![],
        draft: false,
        images: vec![],
        cache: false,
    };
    let (events, _rx) = crate::flow::channel();
    lp.queue.push_back(Queued {
        ticket,
        req: placeholder,
        budget: None,
        events,
        since: Instant::now(),
        deadline,
        score: Some((req, out, deadline)),
    });
    let _ = ack.send(Ok(()));
}
fn expire_scores(engine: &mut dyn ServeEngine, lp: &mut Loop, m: &Metrics) {
    let now = Instant::now();
    let mut keep = VecDeque::new();
    for q in lp.queue.drain(..) {
        if q.score.as_ref().is_some_and(|s| now >= s.2) {
            m.decisions_cancelled.fetch_add(1, Ordering::Relaxed);
            q.fail(SeqError::new(FailKind::RequestTimeout, "decision request deadline exceeded"));
        } else {
            keep.push_back(q);
        }
    }
    lp.queue = keep;
    let ids: Vec<u64> = lp.scores.iter().filter(|(_, s)| now >= s.deadline).map(|(id, _)| *id).collect();
    for id in ids {
        let s = lp.scores.remove(&id).unwrap();
        lp.admitted.remove(&s.ticket);
        engine.cancel(id);
        m.decisions_cancelled.fetch_add(1, Ordering::Relaxed);
        let _ = s.out.send(Err(SeqError::new(FailKind::RequestTimeout, "decision request deadline exceeded")));
    }
}

#[allow(clippy::too_many_arguments)]
fn enqueue(
    engine: &dyn ServeEngine,
    lp: &mut Loop,
    m: &Metrics,
    cfg: &RuntimeConfig,
    ticket: Ticket,
    deadline: Instant,
    req: GenRequest,
    events: StreamSender<SeqMsg>,
    ack: oneshot::Sender<Result<(), Refusal>>,
) {
    m.inbox.fetch_sub(1, Ordering::AcqRel);
    let refuse = |ack: oneshot::Sender<Result<(), Refusal>>, why: Refusal| {
        m.rejected.fetch_add(1, Ordering::Relaxed);
        let _ = ack.send(Err(why));
    };
    if events.is_closed() || ack.is_closed() {
        m.cancelled.fetch_add(1, Ordering::Relaxed);
        let _ = ack.send(Err(Refusal::EngineDown("the reply consumer disconnected".into())));
        return;
    }
    if Instant::now() >= deadline {
        events.fail(SeqError::new(FailKind::RequestTimeout, "the request deadline expired before admission"));
        if !events.has_failure_hook() {
            m.request_timeouts.fetch_add(1, Ordering::Relaxed);
        }
        return refuse(ack, Refusal::RequestTimeout);
    }
    if lp.draining.is_some() {
        return refuse(ack, Refusal::ShuttingDown);
    }
    let req_budget = req.reasoning_budget;
    let req = Request {
        prompt: req.prompt,
        sampling: req.sampling,
        max_new: req.max_new,
        stop: req.stop,
        draft: req.draft,
        images: req.images,
        cache: req.cache,
    };
    let budget = req_budget;
    if let Err(e) = engine.validate(&req) {
        return refuse(ack, Refusal::Rejected(e));
    }
    // a request whose prefill cannot finish before its deadline, even at the backend's optimistic rate and from the longest cached prefix, is
    // refused now instead of holding a slot and the pool for the whole deadline and failing at its end (an engine that states no bound: never)
    if let Some(floor) = engine.prefill_floor(&req) {
        let left = deadline.saturating_duration_since(Instant::now());
        if floor > left {
            m.deadline_refusals.fetch_add(1, Ordering::Relaxed);
            return refuse(ack, Refusal::DeadlineTooShort { floor, left });
        }
    }
    if lp.queue.len() >= cfg.queue_depth {
        m.queue_full.fetch_add(1, Ordering::Relaxed);
        return refuse(ack, Refusal::QueueFull { depth: cfg.queue_depth });
    }
    lp.queue.push_back(Queued { ticket, req, budget, events, since: Instant::now(), deadline, score: None });
    let _ = ack.send(Ok(()));
}

/// Move requests from the front of the queue into the engine while they fit. Strictly in order: a request that does not fit yet
/// holds the line (a later, smaller one never overtakes it).
fn admit(engine: &mut dyn ServeEngine, lp: &mut Loop, m: &Metrics) {
    while let Some(front) = lp.queue.front() {
        match front.score.as_ref().map_or_else(|| engine.would_admit(&front.req), |s| engine.score_would_admit(&s.0)) {
            Ok(()) => {}
            Err(AdmitError::NoSlot | AdmitError::NoMemory { .. }) if engine.busy() => break, // wait for a running sequence to finish
            Err(e) => {
                // idle and still no room (or an error `validate` should have caught): waiting cannot help
                let q = lp.queue.pop_front().unwrap();
                m.rejected.fetch_add(1, Ordering::Relaxed);
                q.fail(SeqError::new(FailKind::Engine, format!("no capacity for this request: {e:?}")));
                continue;
            }
        }
        let q = lp.queue.pop_front().unwrap();
        if let Some((req, out, deadline)) = q.score {
            let plen = req.prompt.len() as u64;
            match engine.score_submit(req) {
                Ok(id) => {
                    m.decisions_admitted.fetch_add(1, Ordering::Relaxed);
                    m.decision_prompt_tokens.fetch_add(plen, Ordering::Relaxed);
                    lp.admitted.insert(q.ticket, id);
                    lp.scores.insert(id, ScoreStream { ticket: q.ticket, out, deadline });
                }
                Err(e) => {
                    m.rejected.fetch_add(1, Ordering::Relaxed);
                    let _ = out.send(Err(SeqError::new(FailKind::Engine, e.to_string())));
                }
            }
            continue;
        }
        let plen = q.req.prompt.len() as u64;
        match engine.submit_deferred_with_budget(q.req, q.budget) {
            Ok(id) => {
                m.observability.queue.record(q.since.elapsed());
                m.requests.fetch_add(1, Ordering::Relaxed);
                m.prompt_tokens.fetch_add(plen, Ordering::Relaxed);
                lp.admitted.insert(q.ticket, id);
                lp.streams.insert(id, Stream { ticket: q.ticket, tx: q.events, deadline: q.deadline });
            }
            Err(e) => {
                m.rejected.fetch_add(1, Ordering::Relaxed);
                let _ = q.events.send(SeqMsg::Done(DoneInfo::failed(SeqError::new(FailKind::Engine, e.to_string()))));
            }
        }
    }
}

/// Answer the requests that have waited past `--queue-timeout` (the queue is in arrival order, so the oldest is in front).
fn expire_requests(engine: &mut dyn ServeEngine, lp: &mut Loop, m: &Metrics) {
    let now = Instant::now();
    lp.queue.retain(|q| {
        if now < q.deadline {
            return true;
        }
        q.events.fail(SeqError::new(FailKind::RequestTimeout, "the request deadline expired while queued"));
        if !q.events.has_failure_hook() {
            m.request_timeouts.fetch_add(1, Ordering::Relaxed);
        }
        false
    });
    let expired: Vec<_> = lp.streams.iter().filter(|(_, s)| now >= s.deadline).map(|(&id, _)| id).collect();
    for id in expired {
        let s = lp.streams.remove(&id).unwrap();
        lp.admitted.remove(&s.ticket);
        s.tx.fail(SeqError::new(FailKind::RequestTimeout, "the request deadline expired"));
        engine.cancel(id);
        if !s.tx.has_failure_hook() {
            m.request_timeouts.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn expire_queue(lp: &mut Loop, m: &Metrics, cfg: &RuntimeConfig) {
    while lp.queue.front().is_some_and(|q| q.since.elapsed() >= cfg.queue_timeout) {
        let q = lp.queue.pop_front().unwrap();
        m.rejected.fetch_add(1, Ordering::Relaxed);
        m.queue_timeouts.fetch_add(1, Ordering::Relaxed);
        q.fail(SeqError::new(
            FailKind::QueueTimeout,
            format!("the request waited {}s for a free slot", cfg.queue_timeout.as_secs()),
        ));
    }
}

fn cancel(engine: &mut dyn ServeEngine, lp: &mut Loop, m: &Metrics, ticket: Ticket) {
    if let Some(at) = lp.queue.iter().position(|q| q.ticket == ticket) {
        let q = lp.queue.remove(at).unwrap();
        if q.score.is_some() {
            m.decisions_cancelled.fetch_add(1, Ordering::Relaxed);
        }
        m.cancelled.fetch_add(1, Ordering::Relaxed);
    } else if let Some(id) = lp.admitted.remove(&ticket) {
        if lp.scores.remove(&id).is_some() {
            m.decisions_cancelled.fetch_add(1, Ordering::Relaxed);
        }
        if engine.cancel(id) {
            m.cancelled.fetch_add(1, Ordering::Relaxed);
            route(engine, lp, m);
        }
        lp.streams.remove(&id);
    }
}

/// Drop requests whose reply nobody is listening for any more. The handlers' `CancelGuard` and the sender-closed checks in their tasks
/// are the fast paths; this is the net under them.
fn reap_closed(engine: &mut dyn ServeEngine, lp: &mut Loop, m: &Metrics) {
    let before = lp.queue.len();
    lp.queue.retain(|q| {
        if q.closed() && q.score.is_some() {
            m.decisions_cancelled.fetch_add(1, Ordering::Relaxed);
        }
        !q.closed()
    });
    let score_gone: Vec<Ticket> = lp.scores.values().filter(|s| s.out.is_closed()).map(|s| s.ticket).collect();
    for t in score_gone {
        cancel(engine, lp, m, t);
    }
    m.cancelled.fetch_add((before - lp.queue.len()) as u64, Ordering::Relaxed);
    let gone: Vec<(u64, Ticket)> = lp.streams.iter().filter(|(_, s)| s.tx.is_closed()).map(|(id, s)| (*id, s.ticket)).collect();
    for (id, ticket) in gone {
        lp.streams.remove(&id);
        lp.admitted.remove(&ticket);
        if engine.cancel(id) {
            m.cancelled.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn start_drain(lp: &mut Loop, m: &Metrics, deadline: Instant) {
    if lp.draining.is_some() {
        return;
    }
    lp.draining = Some(deadline);
    for from in [State::Ready, State::Degraded, State::Loading] {
        m.transition(from, State::Draining);
    }
    let n = lp.queue.len();
    for q in lp.queue.drain(..) {
        m.rejected.fetch_add(1, Ordering::Relaxed);
        q.fail(SeqError::new(FailKind::Shutdown, "the server is shutting down"));
    }
    eprintln!("rexl3 level=info event=draining queued_refused={n} running={}", lp.streams.len());
}

/// Answer everything still waiting or running with `why` (the engine's own state is the caller's business).
fn fail_everything(lp: &mut Loop, why: SeqError) {
    for q in lp.queue.drain(..) {
        q.fail(why.clone());
    }
    for (_, s) in lp.streams.drain() {
        let _ = s.tx.send(SeqMsg::Done(DoneInfo::failed(why.clone())));
    }
    for (_, s) in lp.scores.drain() {
        let _ = s.out.send(Err(why.clone()));
    }
    lp.admitted.clear();
}

/// The engine cannot go on: refuse everything, answer every request with an error, and hand over to `on_fatal`.
fn die(info: &FatalInfo, rx: &Receiver<Cmd>, lp: &mut Loop, m: &Metrics, cfg: &RuntimeConfig) {
    m.set_state(State::Failed);
    eprintln!("rexl3 level=fatal event=engine_failed panicked={} reason=\"{}\"", info.panicked, info.reason);
    let why = SeqError::new(FailKind::EngineDown, format!("the engine failed: {}", info.reason));
    let watch: Vec<StreamSender<SeqMsg>> =
        lp.streams.values().map(|s| s.tx.clone()).chain(lp.queue.iter().map(|q| q.events.clone())).collect();
    fail_everything(lp, why.clone());
    // submits that arrived meanwhile
    while let Ok(c) = rx.try_recv() {
        if let Cmd::Submit { ack, .. } | Cmd::Score { ack, .. } = c {
            m.inbox.fetch_sub(1, Ordering::AcqRel);
            m.rejected.fetch_add(1, Ordering::Relaxed);
            let _ = ack.send(Err(Refusal::EngineDown(info.reason.clone())));
        }
    }
    // Give the HTTP side a moment to write the errors out before `on_fatal` (which normally ends the process): a reply's task drops
    // its receiver once it has seen the end of the stream, and the socket write follows within a few milliseconds.
    let t0 = Instant::now();
    while !watch.iter().all(|t| t.is_closed()) && t0.elapsed() < Duration::from_millis(1500) {
        std::thread::sleep(Duration::from_millis(10));
    }
    if !watch.is_empty() {
        std::thread::sleep(Duration::from_millis(150));
    }
    (cfg.on_fatal)(info);
}

/// Send new tokens to their streams, finish the ones that ended, cancel the ones whose client left.
fn route(engine: &mut dyn ServeEngine, lp: &mut Loop, m: &Metrics) {
    let mut gone = Vec::new();
    for ev in engine.take_events() {
        m.completion_tokens.fetch_add(ev.tokens.len() as u64, Ordering::Relaxed);
        if !ev.tokens.is_empty() {
            m.observability.token_batches.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(s) = lp.streams.get(&ev.id) {
            if !ev.tokens.is_empty() && s.tx.send(SeqMsg::Tokens(ev.tokens)).is_err() {
                gone.push(ev.id);
            }
        }
    }
    for id in gone.drain(..) {
        if let Some(s) = lp.streams.remove(&id) {
            lp.admitted.remove(&s.ticket);
            engine.cancel(id);
            m.cancelled.fetch_add(1, Ordering::Relaxed);
            if !s.tx.has_failure_hook() {
                m.stream_failures.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    for (id, row) in engine.take_scores() {
        if let Some(s) = lp.scores.remove(&id) {
            lp.admitted.remove(&s.ticket);
            let cached = row.cached_tokens as u64;
            if s.out.send(Ok(row)).is_ok() {
                m.decisions_completed.fetch_add(1, Ordering::Relaxed);
                m.decisions_cached_tokens.fetch_add(cached, Ordering::Relaxed);
            } else {
                m.decisions_cancelled.fetch_add(1, Ordering::Relaxed);
                m.cancelled.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    for (id, reason) in engine.take_failed() {
        if let Some(s) = lp.scores.remove(&id) {
            lp.admitted.remove(&s.ticket);
            let _ = s.out.send(Err(SeqError::new(FailKind::Engine, reason.clone())));
        }
        m.failed_sequences.fetch_add(1, Ordering::Relaxed);
        if let Some(s) = lp.streams.remove(&id) {
            lp.admitted.remove(&s.ticket);
            let _ = s.tx.send(SeqMsg::Done(DoneInfo::failed(SeqError::new(FailKind::Engine, reason))));
        }
    }
    for o in engine.take_outputs() {
        m.drafted.fetch_add(o.drafted, Ordering::Relaxed);
        m.accepted.fetch_add(o.accepted, Ordering::Relaxed);
        m.rounds.fetch_add(o.rounds, Ordering::Relaxed);
        if let Some(s) = lp.streams.remove(&o.id) {
            let cancelled = o.finish == Some(gb_engine::FinishReason::Cancelled);
            if !cancelled {
                m.observability.completed.fetch_add(1, Ordering::Relaxed);
            }
            lp.admitted.remove(&s.ticket);
            let _ = s.tx.send(SeqMsg::Done(DoneInfo {
                finish: if cancelled { None } else { o.finish },
                error: None,
                rounds: o.rounds,
                drafted: o.drafted,
                accepted: o.accepted,
                cached_tokens: o.cached_tokens,
            }));
        }
    }
    for id in gone {
        if let Some(s) = lp.streams.remove(&id) {
            lp.admitted.remove(&s.ticket);
            if engine.cancel(id) {
                m.cancelled.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    // cancelling produced final events and outputs; nobody is listening for those, drop them
    let _ = engine.take_events();
    let _ = engine.take_outputs();
}
