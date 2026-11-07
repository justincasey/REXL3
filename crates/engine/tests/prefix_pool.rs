//! The prefix cache of a GLM group (`--prefix-cache on`) at engine level, on the CPU mock in its GLM shape with a pool cache
//! (`MockBackend::with_pool_cache`): the entries keep idle pages of the KV pool, the backend takes cuts in its batched call and at a prompt's
//! end. What these tests pin:
//!
//! * **The end cut.** A policy of no tail (`CutPolicy::tail == 0`) snapshots the whole prompt after its last slice; the next turn, which
//!   repeats the prompt and adds to it, resumes from all of it and replies as it would uncached (tokens and speculation counters).
//! * **Cache prompts in the batched call.** Prompts that take cuts or resume a restored slot go to `Backend::prefill_slices` together with
//!   the others, and every reply is the uncached one.
//! * **Idle pool pages.** Pages an entry keeps are given back (the entry evicted) as soon as the admitted sequences may need them: a request
//!   that needs most of the pool runs to its end with the cache full of an earlier prompt (the mock's pool panics when it runs out).

use gb_engine::mock::{Fault, MockBackend, PrefixFault, MOCK_PAGE};
use gb_engine::{CutPolicy, Drafter, Engine, EngineConfig, Output, PrefillPolicy, PrefixConfig, Request};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};

const VOCAB: usize = 512;
const SLOTS: usize = 4;

fn admission(slots: usize, kv_tokens: Option<u64>) -> Admission {
    let mm = ModelMemory {
        weights_bytes: 1 << 30,
        kv_bytes_per_token: 12_000,
        state_bytes_per_slot: 141 << 20,
        activation_reserve: 1 << 30,
    };
    let a = Admission::new(64 << 30, 6 << 30, mm, slots).unwrap();
    match kv_tokens {
        Some(t) => a.with_kv_tokens(t),
        None => a,
    }
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        max_rows: 64,
        max_drafts: 7,
        cost: CostModel::from_table(vec![(1, 70.0), (16, 74.0), (32, 110.0), (64, 190.0)]),
        drafting: true,
    }
}

fn no_native_drafter() -> Box<dyn Fn() -> Box<dyn Drafter>> {
    Box::new(|| Box::new(gb_engine::NoDrafts) as Box<dyn Drafter>)
}

/// The recipe's policy: the end cut at the prompt's end.
fn policy() -> PrefixConfig {
    PrefixConfig {
        policy: CutPolicy { min_cut: 64, tail: 0, ghost_min_gain: 256, min_gap: 0 },
        ghost_blocks: 1 << 12,
        ..Default::default()
    }
}

fn mock(slots: usize, pages: usize, cache: bool) -> MockBackend {
    let b = MockBackend::new(VOCAB, slots, Fault::None).with_glm_shape().with_native_draft().with_prefill_slice(300);
    if cache {
        b.with_pool_cache(8, pages, PrefixFault::None)
    } else {
        b.with_paged_kv(pages)
    }
}

fn engine(slots: usize, pages: usize, kv_tokens: Option<u64>, cache: bool) -> Engine<MockBackend> {
    let e = Engine::new(mock(slots, pages, cache), admission(slots, kv_tokens), engine_config(), no_native_drafter())
        .with_prefill_policy(PrefillPolicy::Burst { div: 1, cap: slots })
        .with_prefill_batching(true);
    if cache {
        e.with_prefix_cache(policy())
    } else {
        e
    }
}

fn toks(seed: u64, n: usize) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1 + (x % (VOCAB as u64 - 1)) as u32
        })
        .collect()
}

fn greedy(prompt: Vec<u32>, max_new: usize) -> Request {
    Request {
        prompt,
        sampling: Sampling { seed: 7, temperature: 0.0, top_k: 20, top_p: 0.95, min_p: 0.0 },
        max_new,
        stop: vec![],
        draft: true,
        images: vec![],
        cache: true,
    }
}

fn run(e: &mut Engine<MockBackend>, reqs: Vec<Request>) -> Vec<Output> {
    let ids: Vec<u64> = reqs.into_iter().map(|r| e.submit_deferred(r).unwrap()).collect();
    e.run().unwrap();
    let mut outs = e.take_outputs();
    outs.sort_by_key(|o| ids.iter().position(|&i| i == o.id));
    assert_eq!(outs.len(), ids.len());
    outs
}

/// What the uncached engine (`cache: false` on a backend without a cache) answers `r`.
fn reference(r: &Request, pages: usize, kv_tokens: Option<u64>) -> Output {
    let mut e = engine(SLOTS, pages, kv_tokens, false);
    run(&mut e, vec![Request { cache: false, ..r.clone() }]).remove(0)
}

fn same(a: &Output, b: &Output) {
    assert_eq!(a.tokens, b.tokens, "the tokens differ from the uncached reply");
    assert_eq!((a.finish, a.stopped), (b.finish, b.stopped));
    assert_eq!((a.drafted, a.accepted, a.rounds), (b.drafted, b.accepted, b.rounds), "the speculation counters differ");
}

/// A second turn repeats the first turn's prompt and its reply and adds a question: it resumes from the WHOLE first prompt (the end cut,
/// taken after the prompt's last slice), and replies as it does uncached, drafts included (the drafter's window came back with the hit).
#[test]
fn the_next_turn_resumes_from_the_whole_prompt_and_replies_as_uncached() {
    let pages = SLOTS * 64;
    let mut e = engine(SLOTS, pages, None, true);
    let first = greedy(toks(1, 900), 24);
    let a = run(&mut e, vec![first.clone()]).remove(0);
    same(&a, &reference(&first, pages, None));
    assert_eq!(a.cached_tokens, 0);
    let mut second = first.prompt.clone();
    second.extend(&a.tokens);
    second.extend(toks(2, 40));
    let second = greedy(second, 24);
    let b = run(&mut e, vec![second.clone()]).remove(0);
    assert_eq!(b.cached_tokens, 900, "the whole first prompt is reused");
    same(&b, &reference(&second, pages, None));
    let st = e.prefix_stats();
    assert!(st.hits == 1 && st.inserts >= 2, "{st:?}");
}

/// Prompts that take cuts and a prompt that resumes a restored slot share the backend's batched call, and each replies as it does uncached.
#[test]
fn prompts_that_use_the_cache_share_the_batched_call_and_reply_as_uncached() {
    let pages = SLOTS * 64;
    let mut e = engine(SLOTS, pages, None, true);
    let base = greedy(toks(3, 500), 8);
    run(&mut e, vec![base.clone()]);
    let mut grown = base.prompt.clone();
    grown.extend(toks(4, 120));
    let reqs = vec![greedy(toks(5, 260), 16), greedy(grown, 16), greedy(toks(6, 700), 16)];
    let outs = run(&mut e, reqs.clone());
    // run together, the three draft by a shared row budget: the tokens are the contract (the counters are those of a solo run's only alone)
    for (r, o) in reqs.iter().zip(&outs) {
        assert_eq!(o.tokens, reference(r, pages, None).tokens, "the tokens differ from the uncached reply");
    }
    assert_eq!(outs[1].cached_tokens, 500);
    assert!(e.backend.batched_cache_jobs() >= 3, "cache prompts went to the batched call: {}", e.backend.batched_cache_jobs());
    assert!(e.prefix_stats().inserts >= 4, "{:?}", e.prefix_stats());
}

/// The cache keeps idle pool pages only: with an earlier prompt's entry holding most of a small pool, a request that reserves most of the
/// pool is admitted and runs to its end (its pages come from the evicted entry), and replies as it does uncached.
#[test]
fn the_cache_gives_its_pool_pages_back_to_an_admitted_sequence() {
    // 24 pages: admission hands out 20 pages' tokens (the planner keeps a page a slot beyond them)
    let pages = 24;
    let kv_tokens = Some(((pages - SLOTS) * MOCK_PAGE) as u64);
    let mut e = engine(SLOTS, pages, kv_tokens, true);
    let first = greedy(toks(7, 14 * MOCK_PAGE), 8);
    run(&mut e, vec![first]);
    assert!(e.prefix_stats().pages_held >= 14, "{:?}", e.prefix_stats());
    let big = greedy(toks(8, 17 * MOCK_PAGE), 16);
    let out = run(&mut e, vec![big.clone()]).remove(0);
    same(&out, &reference(&big, pages, kv_tokens));
    let st = e.prefix_stats();
    assert!(st.evicted_for_pool >= 1, "the entry gave its pages back: {st:?}");
}
