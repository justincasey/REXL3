//! The slices a burst runs in one step reach the backend as ONE `prefill_slices` call (`Engine::with_prefill_batching`), and every prompt in
//! it is treated as it would be alone: its own first token, its own failure, its own place in the queue.

use gb_engine::mock::{ChainDrafter, Fault, MockBackend};
use gb_engine::{
    Backend, BackendError, CutStep, Drafter, Engine, EngineConfig, ImageSpan, PrefillJob, PrefillPolicy, PrefillStep, Request,
    RowOutput, RowRequest, StepError,
};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

const VOCAB: usize = 512;

#[derive(Default)]
struct Log {
    /// Jobs in each `prefill_slices` call.
    batches: Vec<usize>,
    /// `prefill_slice` calls made one by one.
    singles: usize,
}

struct Batching {
    inner: MockBackend,
    log: Rc<RefCell<Log>>,
    /// A prompt that starts with this token cannot be served: its row is NaN and the call reports a recoverable error.
    poison: Option<u32>,
    /// Every call of `prefill_slices` ends in a fatal error.
    fatal: bool,
    error: Option<BackendError>,
}

impl Backend for Batching {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn take_error(&mut self) -> Option<BackendError> {
        self.error.take()
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
        self.log.borrow_mut().singles += 1;
        self.inner.prefill_slice(slot, prompt, images, done, candidates)
    }
    fn prefill_slices(&mut self, jobs: &[PrefillJob<'_>]) -> Vec<CutStep> {
        self.log.borrow_mut().batches.push(jobs.len());
        if self.fatal {
            self.error = Some(BackendError::fatal("injected fatal error in a batched prefill"));
        }
        jobs.iter()
            .map(|j| {
                let mut step = self.inner.prefill_slice(j.slot, j.prompt, j.images, j.done, j.candidates);
                if self.poison == j.prompt.first().copied() && self.poison.is_some() {
                    if let Some(row) = step.row.as_mut() {
                        row.values.iter_mut().for_each(|v| *v = f32::NAN);
                    }
                    self.error.get_or_insert(BackendError::retryable("injected failure of one prompt"));
                }
                CutStep { step, captured: Vec::new() }
            })
            .collect()
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        self.inner.step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.inner.commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
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

struct Setup {
    slice: usize,
    poison: Option<u32>,
    fatal: bool,
    batching: bool,
    policy: PrefillPolicy,
}

impl Default for Setup {
    fn default() -> Self {
        Setup { slice: 0, poison: None, fatal: false, batching: true, policy: PrefillPolicy::Burst { div: 4, cap: 8 } }
    }
}

fn engine(set: Setup, log: &Rc<RefCell<Log>>) -> Engine<Batching> {
    let mut inner = MockBackend::new(VOCAB, 40, Fault::None);
    if set.slice > 0 {
        inner = inner.with_prefill_slice(set.slice);
    }
    Engine::new(
        Batching { inner, log: log.clone(), poison: set.poison, fatal: set.fatal, error: None },
        admission(40),
        EngineConfig { max_rows: 64, max_drafts: 7, cost: CostModel::from_table(vec![(1, 70.0), (64, 80.0)]), drafting: true },
        Box::new(|| Box::new(ChainDrafter { vocab: VOCAB, corrupt_every: 5 }) as Box<dyn Drafter>),
    )
    .with_prefill_policy(set.policy)
    .with_prefill_batching(set.batching)
}

fn request(i: usize, len: usize) -> Request {
    Request {
        // the first token is distinct per request, so a test can single one out (`Setup::poison`)
        prompt: std::iter::once(300 + i as u32).chain((1..len).map(|j| ((i * 97 + j * 13 + 5) % 250) as u32)).collect(),
        sampling: Sampling {
            seed: 900 + i as u64,
            temperature: if i % 2 == 0 { 0.0 } else { 0.9 },
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
        },
        max_new: 30,
        stop: vec![],
        draft: true,
        images: vec![],
        cache: true,
    }
}

fn submit(e: &mut Engine<Batching>, n: usize, len_of: impl Fn(usize) -> usize) -> Vec<u64> {
    (0..n).map(|i| e.submit_deferred(request(i, len_of(i))).unwrap()).collect()
}

fn outputs(e: &mut Engine<Batching>) -> BTreeMap<u64, Vec<u32>> {
    e.take_outputs().into_iter().map(|o| (o.id, o.tokens)).collect()
}

#[test]
fn a_burst_is_one_backend_call_a_step_and_every_reply_equals_the_unbatched_one() {
    let (log_b, log_s) = (Rc::new(RefCell::new(Log::default())), Rc::new(RefCell::new(Log::default())));
    let mut batched = engine(Setup::default(), &log_b);
    submit(&mut batched, 32, |i| 6 + i % 11);
    batched.run().unwrap();
    let got = outputs(&mut batched);
    // the same burst, one slice at a time (batching off) and one slice a step (serial): the same tokens
    let mut single = engine(Setup { batching: false, ..Default::default() }, &log_s);
    submit(&mut single, 32, |i| 6 + i % 11);
    single.run().unwrap();
    assert_eq!(outputs(&mut single), got);
    let mut serial =
        engine(Setup { policy: PrefillPolicy::Serial, ..Default::default() }, &Rc::new(RefCell::new(Log::default())));
    submit(&mut serial, 32, |i| 6 + i % 11);
    serial.run().unwrap();
    assert_eq!(outputs(&mut serial), got);
    assert_eq!(got.len(), 32);
    // 32 waiting: 8 slices in the first step, then ceil(24/4) = 6, 5 (18 left), 4 (13), 3 (9), 2 (6), then one a step: calls of 8, 6, 5, 4, 3, 2 jobs
    let b = log_b.borrow();
    assert_eq!(b.batches, vec![8, 6, 5, 4, 3, 2], "{:?}", b.batches);
    assert_eq!(b.batches.iter().sum::<usize>() + b.singles, 32);
    assert_eq!(log_s.borrow().batches.len(), 0, "with batching off the backend is only asked one slice at a time");
}

#[test]
fn the_shipped_qwen_policy_batches_from_three_waiting_prompts_and_never_for_fewer() {
    assert_eq!(PrefillPolicy::BATCHED_SHORT, PrefillPolicy::Burst { div: 2, cap: 16 });
    for (waiting, first_call) in [(1usize, None), (2, None), (3, Some(2usize)), (8, Some(4)), (32, Some(16))] {
        let log = Rc::new(RefCell::new(Log::default()));
        let mut e = engine(Setup { policy: PrefillPolicy::BATCHED_SHORT, ..Default::default() }, &log);
        submit(&mut e, waiting, |i| 8 + i % 7);
        e.step().unwrap();
        assert_eq!(log.borrow().batches.first().copied(), first_call, "{waiting} waiting prompts");
    }
}

#[test]
fn an_engine_that_does_not_opt_in_prefills_one_slice_a_step_and_never_batches() {
    // the policy is a family's choice: `Engine::new` alone is the first release's serial prefill, whatever else a family's backend can do
    assert_eq!(PrefillPolicy::default(), PrefillPolicy::Serial);
    let log = Rc::new(RefCell::new(Log::default()));
    let mut e = engine(Setup { policy: PrefillPolicy::default(), ..Default::default() }, &log);
    submit(&mut e, 12, |i| 8 + i % 7);
    for step in 1..=4 {
        e.step().unwrap();
        let l = log.borrow();
        assert!(l.batches.is_empty(), "step {step}: a batch call under the serial policy: {:?}", l.batches);
        assert_eq!(l.singles, step, "step {step}: one slice a step");
    }
}

#[test]
fn a_queue_of_long_prompts_gets_one_chunk_a_step_and_a_few_hundred_rows_each_share_one() {
    // 6,000-row prompts: one slice already counts a whole step's budget of 4,096 rows, so 12 of them are served one per step (as before)
    let log = Rc::new(RefCell::new(Log::default()));
    let mut e = engine(Setup { slice: 2000, ..Default::default() }, &log);
    submit(&mut e, 12, |_| 6000);
    for _ in 0..5 {
        e.step().unwrap();
    }
    {
        let l = log.borrow();
        assert_eq!(
            l.batches.iter().sum::<usize>() + l.singles,
            5,
            "five steps, one slice each: {:?} and {} singles",
            l.batches,
            l.singles
        );
        assert!(l.batches.iter().all(|&n| n == 1), "{:?}", l.batches);
    }
    // 900-row prompts, 20 waiting (five slices by the count): four fit a step's 4,096 rows (a fifth would make 4,500)
    let log = Rc::new(RefCell::new(Log::default()));
    let mut e = engine(Setup { slice: 900, ..Default::default() }, &log);
    submit(&mut e, 20, |_| 900);
    e.step().unwrap();
    assert_eq!(log.borrow().batches, vec![4], "{:?}", log.borrow().batches);
}

#[test]
fn a_lone_prompt_or_a_short_queue_never_makes_a_batch_call() {
    let log = Rc::new(RefCell::new(Log::default()));
    let mut e = engine(Setup::default(), &log);
    submit(&mut e, 1, |_| 12);
    e.run().unwrap();
    e.take_outputs();
    submit(&mut e, 4, |_| 12);
    e.run().unwrap();
    assert!(log.borrow().batches.is_empty(), "{:?}", log.borrow().batches);
    assert_eq!(log.borrow().singles, 5);
}

#[test]
fn a_prompt_the_backend_cannot_serve_fails_alone_and_the_others_get_their_own_replies() {
    // ten prompts; the one that starts with token 303 (request 3) cannot be served
    let log = Rc::new(RefCell::new(Log::default()));
    let mut e = engine(Setup { poison: Some(303), ..Default::default() }, &log);
    let ids = submit(&mut e, 10, |i| 8 + i);
    let free = e.admission().capacity();
    e.run().unwrap();
    let failed = e.take_failed();
    assert_eq!(failed.len(), 1, "{failed:?}");
    assert_eq!(failed[0].0, ids[3]);
    assert!(failed[0].1.contains("injected failure"), "{}", failed[0].1);
    let got = outputs(&mut e);
    assert_eq!(got.len(), 9);
    assert!(!got.contains_key(&ids[3]));
    assert_eq!(e.admission().available(), free, "the failed prompt's slot is back too");
    // each of the nine equals its solo run
    for i in (0..10).filter(|&i| i != 3) {
        let mut solo =
            engine(Setup { policy: PrefillPolicy::Serial, ..Default::default() }, &Rc::new(RefCell::new(Log::default())));
        let id = solo.submit_deferred(request(i, 8 + i)).unwrap();
        solo.run().unwrap();
        assert_eq!(outputs(&mut solo)[&id], got[&ids[i]], "request {i}");
    }
}

#[test]
fn a_fatal_error_stops_the_step_and_keeps_every_prompt_queued_in_order() {
    let log = Rc::new(RefCell::new(Log::default()));
    let mut e = engine(Setup { fatal: true, ..Default::default() }, &log);
    submit(&mut e, 12, |i| 8 + i % 5);
    let err = e.step().unwrap_err();
    assert!(matches!(err, StepError::Fatal(_)), "{err}");
    assert_eq!(e.prefilling(), 12, "no prompt was lost or failed by a fatal backend error");
    assert_eq!(e.active(), 0);
}

#[test]
fn prompts_that_need_more_than_one_slice_go_back_in_turn_and_finish_with_the_same_tokens() {
    // 24-token prompts in slices of 8: three slices each, in one call a step while the burst lasts
    let log = Rc::new(RefCell::new(Log::default()));
    let mut e = engine(Setup { slice: 8, ..Default::default() }, &log);
    submit(&mut e, 16, |_| 24);
    e.run().unwrap();
    let got = outputs(&mut e);
    assert_eq!(got.len(), 16);
    let mut serial =
        engine(Setup { slice: 8, policy: PrefillPolicy::Serial, ..Default::default() }, &Rc::new(RefCell::new(Log::default())));
    submit(&mut serial, 16, |_| 24);
    serial.run().unwrap();
    assert_eq!(outputs(&mut serial), got);
    assert!(log.borrow().batches.iter().all(|&n| n >= 2), "{:?}", log.borrow().batches);
    let l = log.borrow();
    assert_eq!(
        l.batches.iter().sum::<usize>() + l.singles,
        48,
        "every one of the 48 slices went through a batch call or a single one"
    );
    assert!(
        l.batches.iter().sum::<usize>() > l.singles,
        "the burst is mostly batched: {:?} and {} singles",
        l.batches,
        l.singles
    );
}
