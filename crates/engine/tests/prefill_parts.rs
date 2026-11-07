//! A backend that runs a long prompt slice in parts while sequences decode (`Backend::prefill_in_flight`: GLM's tensor-parallel rank 0,
//! `tp_part`): the engine gives the prompt in flight its next part before any other prompt whose rows would not fit the backend's room,
//! lets prompts that fit run between two parts, and never asks for another long prompt's slice meanwhile. Only when slices run moves:
//! every reply equals its solo run.

use gb_engine::mock::{ChainDrafter, Fault, MockBackend};
use gb_engine::{
    Backend, CutStep, Drafter, Engine, EngineConfig, ImageSpan, PrefillJob, PrefillPolicy, PrefillStep, Request, RowOutput,
    RowRequest,
};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

const VOCAB: usize = 512;
/// A slice of at least these rows runs in parts while a sequence decodes.
const LONG: usize = 100;
/// Calls a slice in parts takes.
const PARTS: usize = 4;
/// Rows another prompt's slice may have while one runs in parts.
const ROOM: usize = 32;

/// What the backend saw, in order: `begin s`, `part s`, `last s` (a slice in parts), `between s` (a slice run while one is in flight),
/// `whole s` (a slice alone), `refused s` (a slice asked for while one is in flight that does not fit the room).
type Log = Rc<RefCell<Vec<String>>>;

struct Parts {
    inner: MockBackend,
    log: Log,
    decode_waiting: bool,
    /// Calls a slice in parts takes.
    parts: usize,
    /// The slot in flight and the calls left of its slice.
    inflight: Option<(usize, usize)>,
}

impl Parts {
    fn note(&self, what: &str, slot: usize) {
        self.log.borrow_mut().push(format!("{what} {slot}"));
    }
}

impl Backend for Parts {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn set_decode_waiting(&mut self, waiting: bool) {
        self.decode_waiting = waiting;
    }
    fn prefill_in_flight(&self) -> Option<(usize, usize)> {
        self.inflight.map(|(s, _)| (s, ROOM))
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.inner.prefill(slot, start, tokens, candidates)
    }
    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> PrefillStep {
        match self.inflight {
            Some((s, left)) if s == slot => {
                if left > 1 {
                    self.inflight = Some((s, left - 1));
                    self.note("part", slot);
                    return PrefillStep { done, row: None };
                }
                self.inflight = None;
                self.note("last", slot);
            }
            Some(_) if prompt.len() - done > ROOM => {
                self.note("refused", slot);
                return PrefillStep { done, row: None };
            }
            Some(_) => self.note("between", slot),
            None if self.decode_waiting && prompt.len() - done >= LONG => {
                self.inflight = Some((slot, self.parts - 1));
                self.note("begin", slot);
                return PrefillStep { done, row: None };
            }
            None => self.note("whole", slot),
        }
        self.inner.prefill_slice(slot, prompt, images, done, candidates)
    }
    fn prefill_slices(&mut self, jobs: &[PrefillJob<'_>]) -> Vec<CutStep> {
        jobs.iter()
            .map(|j| CutStep { step: self.prefill_slice(j.slot, j.prompt, j.images, j.done, j.candidates), captured: Vec::new() })
            .collect()
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        self.inner.step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.inner.commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        if self.inflight.is_some_and(|(s, _)| s == slot) {
            self.inflight = None;
        }
        self.inner.release(slot)
    }
}

fn admission(slots: usize) -> Admission {
    let mm = ModelMemory {
        weights_bytes: 1 << 30,
        kv_bytes_per_token: 65536,
        state_bytes_per_slot: 150 << 20,
        activation_reserve: 1 << 30,
    };
    Admission::new(100 << 30, 6 << 30, mm, slots).unwrap()
}

fn config() -> EngineConfig {
    EngineConfig { max_rows: 64, max_drafts: 7, cost: CostModel::from_table(vec![(1, 70.0), (64, 80.0)]), drafting: true }
}

fn drafter() -> Box<dyn Fn() -> Box<dyn Drafter>> {
    Box::new(|| Box::new(ChainDrafter { vocab: VOCAB, corrupt_every: 5 }) as Box<dyn Drafter>)
}

fn engine(policy: PrefillPolicy, log: &Log) -> Engine<Parts> {
    engine_with(MockBackend::new(VOCAB, 8, Fault::None), PARTS, policy, log)
}

/// [`engine`] over `inner`, with slices in parts of `parts` calls.
fn engine_with(inner: MockBackend, parts: usize, policy: PrefillPolicy, log: &Log) -> Engine<Parts> {
    Engine::new(Parts { inner, log: log.clone(), decode_waiting: false, parts, inflight: None }, admission(8), config(), drafter())
        .with_prefill_policy(policy)
        .with_prefill_batching(true)
}

fn request(i: usize, len: usize, max_new: usize) -> Request {
    Request {
        prompt: std::iter::once(300 + i as u32).chain((1..len).map(|j| ((i * 97 + j * 13 + 5) % 250) as u32)).collect(),
        sampling: Sampling { seed: 900 + i as u64, temperature: if i % 2 == 0 { 0.0 } else { 0.9 }, top_k: 20, top_p: 0.95, min_p: 0.0 },
        max_new,
        stop: vec![],
        draft: true,
        images: vec![],
        cache: true,
    }
}

/// Request `i`'s reply alone, on the plain mock (serial prefill, no parts).
fn solo(i: usize, len: usize, max_new: usize) -> Vec<u32> {
    let mut e = Engine::new(MockBackend::new(VOCAB, 8, Fault::None), admission(8), config(), drafter());
    let id = e.submit_deferred(request(i, len, max_new)).unwrap();
    e.run().unwrap();
    e.take_outputs().into_iter().find(|o| o.id == id).unwrap().tokens
}

/// Two chats decode; a long prompt arrives (it runs in parts), then a short one and another long one while it is in flight. Returns the
/// backend's log and every reply by request.
fn scenario(policy: PrefillPolicy) -> (Vec<String>, BTreeMap<usize, Vec<u32>>) {
    let log: Log = Rc::default();
    let mut e = engine(policy, &log);
    let shapes = [(0usize, 12usize, 60usize), (1, 9, 60), (2, 300, 8), (3, 10, 8), (4, 200, 8)];
    let mut ids = BTreeMap::new();
    for &(i, len, max_new) in &shapes[..2] {
        ids.insert(e.submit_deferred(request(i, len, max_new)).unwrap(), i);
    }
    for _ in 0..3 {
        e.step().unwrap();
    }
    let (i, len, max_new) = shapes[2];
    ids.insert(e.submit_deferred(request(i, len, max_new)).unwrap(), i);
    e.step().unwrap();
    for &(i, len, max_new) in &shapes[3..] {
        ids.insert(e.submit_deferred(request(i, len, max_new)).unwrap(), i);
    }
    e.run().unwrap();
    let got: BTreeMap<usize, Vec<u32>> = e.take_outputs().into_iter().map(|o| (ids[&o.id], o.tokens)).collect();
    for &(i, len, max_new) in &shapes {
        assert_eq!(got[&i], solo(i, len, max_new), "request {i}");
    }
    let l = log.borrow().clone();
    (l, got)
}

#[test]
fn the_slice_in_parts_goes_first_a_short_prompt_runs_between_its_parts_and_no_long_one_is_asked_for() {
    let (log, _) = scenario(PrefillPolicy::Burst { div: 1, cap: 8 });
    assert!(!log.iter().any(|l| l.starts_with("refused")), "a long prompt was asked for while one was in flight: {log:?}");
    let at = |what: &str| log.iter().position(|l| l == what).unwrap_or_else(|| panic!("no `{what}` in {log:?}"));
    // the long prompt of slot 2 began in parts while the chats decoded; the short one (slot 3) ran between two of its parts; the second
    // long one (slot 4) began only after the first one's last part
    assert!(at("begin 2") < at("between 3") && at("between 3") < at("last 2"), "{log:?}");
    assert!(at("last 2") < at("begin 4") || at("last 2") < at("whole 4"), "{log:?}");
    assert_eq!(log.iter().filter(|l| l.ends_with(" 2") && (l.starts_with("part") || l.starts_with("last"))).count(), PARTS - 1);
}

#[test]
fn under_the_serial_policy_the_slice_in_parts_is_served_every_step_until_it_ends() {
    let (log, _) = scenario(PrefillPolicy::Serial);
    assert!(!log.iter().any(|l| l.starts_with("refused") || l.starts_with("between")), "{log:?}");
    let begin = log.iter().position(|l| l == "begin 2").unwrap();
    // the parts of slot 2 follow one another: nothing else is prefilled until its last
    assert_eq!(&log[begin..begin + PARTS], &["begin 2", "part 2", "part 2", "last 2"][..], "{log:?}");
}

/// While a slice runs in parts its parts are ONE slice to the pass-over guard: a long prompt that waits behind it (it does not fit the
/// room) is not passed over by each of them. So when a slice ends, the prompt with the fewest rows left goes next, and two long prompts
/// do not take turns slice by slice (each part counted, the one behind would pass the guard before every slice of the other ends).
#[test]
fn a_prompt_waiting_behind_a_slice_in_parts_is_not_passed_over_by_each_of_its_parts() {
    let log: Log = Rc::default();
    // slices of 100 rows, each in more parts than the guard's steps
    let parts = gb_engine::PREFILL_PASS_GUARD as usize + 4;
    let inner = MockBackend::new(VOCAB, 8, Fault::None).with_prefill_slice(100);
    let mut e = engine_with(inner, parts, PrefillPolicy::BATCHED_SHORT, &log);
    // two chats that decode throughout; a 400-row prompt (four slices) begins in parts, then a 200-row one (two slices) arrives
    let shapes = [(0usize, 12usize, 2000usize), (1, 9, 2000), (2, 400, 4), (3, 200, 4)];
    let mut ids = BTreeMap::new();
    for &(i, len, max_new) in &shapes[..2] {
        ids.insert(e.submit_deferred(request(i, len, max_new)).unwrap(), i);
    }
    for _ in 0..3 {
        e.step().unwrap();
    }
    for &(i, len, max_new) in &shapes[2..] {
        ids.insert(e.submit_deferred(request(i, len, max_new)).unwrap(), i);
        e.step().unwrap();
    }
    e.run().unwrap();
    let got: BTreeMap<usize, Vec<u32>> = e.take_outputs().into_iter().map(|o| (ids[&o.id], o.tokens)).collect();
    for &(i, len, max_new) in &shapes {
        assert_eq!(got[&i], solo(i, len, max_new), "request {i}");
    }
    let log = log.borrow().clone();
    let begins: Vec<&str> = log.iter().filter(|l| l.starts_with("begin")).map(String::as_str).collect();
    // slot 2's first slice was in flight when slot 3 arrived; then slot 3 (fewer rows left) to its end, then the rest of slot 2
    assert_eq!(begins, ["begin 2", "begin 3", "begin 3", "begin 2", "begin 2", "begin 2"], "{log:?}");
}
