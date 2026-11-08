//! The NextN (MTP) drafter of GLM-5.3-Flash on the lockstep tensor-parallel seam (`docs/GLM_TP.md` section 9, lane L12).
//!
//! **The calls.** A draft round is two family calls beside the core ones: [`GlmCall::MtpDraft`] (kind [`K_MTP_DRAFT`]) carries what rank
//! 0's speculation runtime decided, every ask's slot, the sequence's tokens and the DEPTH (the arbiter's choice in `--spec auto`, the fixed
//! depth in `--spec mtp`), and every ask's sampler (the keyed-noise seed, temperature, `top_k`, `top_p`, `min_p`, or none for the argmax).
//! The follower applies it with `GlmModel::mtp_draft_keyed` over those arguments and never runs the arbiter, never reads a clock or
//! `REXL3_MTP_DEPTH` / `REXL3_GLM_DRAFT_NOISE`. Its ack carries [`drafts_digest`] (FNV-1a 64 of every ask's presence, the drafted ids and the
//! bits of their values), compared with rank 0's: a draft that differs loses the group naming `mtp_draft`. [`GlmCall::MtpDisable`]
//! (kind [`K_MTP_DISABLE`]) switches a slot's drafting off on every rank.
//!
//! **The NextN layer on a rank.** Its routed experts are OWNED like the trunk's (a rank maps, reads and holds only its share, bound as
//! subbanks; or the whole bank masked to the share on L2b's whole-pack path), so every forward of the layer (each prefill chunk's, the
//! absorb's and each further draft depth's) runs this rank's pairs, ONE all-reduce of the pair buffer `y_pairs` (`rows * 8 * 4096` f32), and
//! the single-node combine with the shared expert last (`model/mtp.rs` `mtp_block`, `model/tp.rs`). The number of those forwards and their
//! rows are functions of the call's arguments and the slots' books (host state moved by the call stream alone), so every rank issues the
//! same collectives with the same counts. The draft head (the trunk `lm_head`, or its restricted copy over the same strips: the
//! `mtp_vocab` fact) is replicated.
//!
//! **The backend half.** `SpecRuntime::propose` is generic over [`DraftModel`]: one node passes the `GlmModel`, rank 0 its `TpLeader`,
//! whose `mtp_draft_keyed` forwards the call before it runs it here and whose `flush` is the core call (so the leader's stream is exactly
//! the single node's list of model calls).

use super::model::{DflashAsk, DflashReq, DraftAsk, Drafts, GlmModel, TapBook};
use crate::draft::Picked;
use super::tp::GlmCall;
use crate::error::ModelError;
use crate::lockstep::{CallReply, Dec, Enc, LockstepRuntime, TpLeader, WireError};
use gb_sampling::Sampling;

/// The frame kind of [`GlmCall::MtpDraft`] (the first of the family's own kinds, `32..=127`).
pub const K_MTP_DRAFT: u8 = 32;
/// The frame kind of [`GlmCall::MtpDisable`].
pub const K_MTP_DISABLE: u8 = 33;
/// The frame kind of [`GlmCall::DflashDraft`]: a round of the DFlash2 drafter split over the group.
pub const K_DFLASH_DRAFT: u8 = 34;
/// The frame kind of [`GlmCall::DflashPromptEnd`].
pub const K_DFLASH_PROMPT_END: u8 = 35;

/// One ask of a draft call on the wire (`DraftAsk` owns its tokens here).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AskWire {
    pub slot: u32,
    /// Drafts wanted (0 only absorbs): rank 0's arbiter or fixed depth decided it.
    pub depth: u32,
    pub tokens: Vec<u32>,
}

impl AskWire {
    pub fn of(a: &DraftAsk<'_>) -> AskWire {
        AskWire { slot: a.slot as u32, depth: a.depth as u32, tokens: a.tokens.to_vec() }
    }
    pub fn ask(&self) -> DraftAsk<'_> {
        DraftAsk { slot: self.slot as usize, tokens: &self.tokens, depth: self.depth as usize }
    }
}

fn enc_sampling(e: &mut Enc, s: &Option<Sampling>) {
    match s {
        Some(s) => {
            e.bool(true);
            e.u64(s.seed);
            e.f64(s.temperature);
            e.u64(s.top_k as u64);
            e.f64(s.top_p);
            e.f64(s.min_p);
        }
        None => e.bool(false),
    }
}

fn dec_sampling(d: &mut Dec<'_>) -> Result<Option<Sampling>, WireError> {
    if !d.bool("glm sampler tag")? {
        return Ok(None);
    }
    Ok(Some(Sampling {
        seed: d.u64("glm sampler seed")?,
        temperature: d.f64("glm sampler temperature")?,
        top_k: usize::try_from(d.u64("glm sampler top_k")?).map_err(|_| WireError::Invalid("top_k overflow".into()))?,
        top_p: d.f64("glm sampler top_p")?,
        min_p: d.f64("glm sampler min_p")?,
    }))
}

/// The body of [`GlmCall::MtpDraft`]: the asks (slot, depth, tokens), then one sampler per ask (or none at all: `keyed` may be empty).
pub(super) fn encode_draft(e: &mut Enc, asks: &[AskWire], keyed: &[Option<Sampling>]) {
    e.u64(asks.len() as u64);
    for a in asks {
        e.u32(a.slot);
        e.u32(a.depth);
        e.u32s(&a.tokens);
    }
    e.u64(keyed.len() as u64);
    for k in keyed {
        enc_sampling(e, k);
    }
}

pub(super) fn decode_draft(d: &mut Dec<'_>) -> Result<GlmCall, WireError> {
    let n = d.len(16, "glm draft asks")?;
    let mut asks = Vec::with_capacity(n);
    for _ in 0..n {
        asks.push(AskWire { slot: d.u32("glm ask slot")?, depth: d.u32("glm ask depth")?, tokens: d.u32s("glm ask tokens")? });
    }
    let k = d.len(1, "glm draft samplers")?;
    let mut keyed = Vec::with_capacity(k);
    for _ in 0..k {
        keyed.push(dec_sampling(d)?);
    }
    Ok(GlmCall::MtpDraft { asks, keyed })
}

/// The body of [`GlmCall::DflashDraft`]: per request its slot, position, anchor, drafts wanted, coverage (f32 bits) and sampler.
pub(super) fn encode_dflash(e: &mut Enc, reqs: &[DflashReq]) {
    e.u64(reqs.len() as u64);
    for r in reqs {
        e.u32(r.slot as u32);
        e.u32(r.p as u32);
        e.u32(r.anchor);
        e.u32(r.want as u32);
        e.u32(r.coverage.to_bits());
        enc_sampling(e, &Some(r.sampling));
    }
}

pub(super) fn decode_dflash(d: &mut Dec<'_>) -> Result<GlmCall, WireError> {
    let n = d.len(21, "glm dflash requests")?;
    let mut reqs = Vec::with_capacity(n);
    for _ in 0..n {
        let (slot, p, anchor, want) =
            (d.u32("glm dflash slot")?, d.u32("glm dflash position")?, d.u32("glm dflash anchor")?, d.u32("glm dflash want")?);
        let coverage = f32::from_bits(d.u32("glm dflash coverage")?);
        let sampling = dec_sampling(d)?.ok_or_else(|| WireError::Invalid("a dflash request without its sampler".into()))?;
        reqs.push(DflashReq { slot: slot as usize, p: p as usize, anchor, want: want as usize, coverage, sampling });
    }
    Ok(GlmCall::DflashDraft { reqs })
}

/// What a split draft round's ack carries: FNV-1a 64 over every sequence's picks (the count, each token, the bits of each score and
/// margin). Both ranks of a split drafter hold the same picks; a rank that drafted anything else loses the group naming `dflash_draft`.
pub fn picks_digest(p: &[Picked]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    };
    eat(b"glm-dflash-picks-v1");
    eat(&(p.len() as u64).to_le_bytes());
    for x in p {
        eat(&(x.tokens.len() as u64).to_le_bytes());
        for (i, &t) in x.tokens.iter().enumerate() {
            eat(&t.to_le_bytes());
            eat(&x.score.get(i).copied().unwrap_or(f32::NAN).to_bits().to_le_bytes());
            eat(&x.margin.get(i).copied().unwrap_or(f32::NAN).to_bits().to_le_bytes());
        }
    }
    h
}

/// What a draft call's ack carries: FNV-1a 64 over every ask's result (a presence byte; then the count, each drafted id and the bits of each
/// value, little-endian). Two ranks that drafted anything differently, or switched a slot off where the other did not, differ here.
pub fn drafts_digest(d: &[Option<Drafts>]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    };
    eat(b"glm-drafts-v1");
    eat(&(d.len() as u64).to_le_bytes());
    for x in d {
        match x {
            None => eat(&[0]),
            Some(x) => {
                eat(&[1]);
                eat(&(x.ids.len() as u64).to_le_bytes());
                for &i in &x.ids {
                    eat(&i.to_le_bytes());
                }
                eat(&(x.vals.len() as u64).to_le_bytes());
                for &v in &x.vals {
                    eat(&v.to_bits().to_le_bytes());
                }
            }
        }
    }
    h
}

/// The `mtp_vocab` fact: `none` without the NextN layer, `full` for the whole head, else the restricted head's prefix strips, the share of
/// the adaptive mode (0: restricted in every call) and the FNV-1a 64 of the strip ranges it is built from (`model::vocab_ranges`), so two
/// ranks that would build another head, or choose it in other calls, are refused by name.
pub fn mtp_vocab_fact(mtp: bool, setting: Option<(usize, f32)>, vocab: usize) -> String {
    if !mtp {
        return "none".into();
    }
    let Some((strips, share)) = setting else { return "full".into() };
    let ranges = super::model::vocab_ranges(strips, vocab);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for (lo, hi) in &ranges {
        for b in (*lo as u64).to_le_bytes().into_iter().chain((*hi as u64).to_le_bytes()) {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    format!("strips={strips} share={share} ranges={h:016x}")
}

/// What the speculation runtime needs from the model it drafts with (`SpecRuntime::propose`): one node's `GlmModel`, or rank 0's
/// `TpLeader`, which forwards the state-changing calls first.
pub trait DraftModel {
    fn has_mtp(&self) -> bool;
    /// The pending commits applied before a draft round (the core `flush`; a model's KDA commits may still run beside the round:
    /// `GlmModel::flush_lazy`, joined by the next forward).
    fn flush(&mut self) -> Result<(), ModelError>;
    /// The slot's tap book (host state, identical on every rank).
    fn mtp_book(&self, slot: usize) -> Option<TapBook>;
    fn mtp_draft_keyed(&mut self, asks: &[DraftAsk<'_>], keyed: &[Option<Sampling>]) -> Result<Vec<Option<Drafts>>, ModelError>;
    /// Whether the model drafts with DFlash2 (`--draft DIR`), which then takes the NextN head's place.
    fn has_dflash(&self) -> bool;
    /// A DFlash2 round (`GlmModel::dflash_draft`): the pending commits must have been applied.
    fn dflash_draft(&mut self, asks: &[DflashAsk<'_>]) -> Result<Vec<Picked>, ModelError>;
}

impl DraftModel for GlmModel {
    fn has_mtp(&self) -> bool {
        GlmModel::has_mtp(self)
    }
    fn flush(&mut self) -> Result<(), ModelError> {
        GlmModel::flush_lazy(self)
    }
    fn mtp_book(&self, slot: usize) -> Option<TapBook> {
        GlmModel::mtp_book(self, slot).cloned()
    }
    fn mtp_draft_keyed(&mut self, asks: &[DraftAsk<'_>], keyed: &[Option<Sampling>]) -> Result<Vec<Option<Drafts>>, ModelError> {
        GlmModel::mtp_draft_keyed(self, asks, keyed)
    }
    fn has_dflash(&self) -> bool {
        GlmModel::has_dflash(self)
    }
    fn dflash_draft(&mut self, asks: &[DflashAsk<'_>]) -> Result<Vec<Picked>, ModelError> {
        GlmModel::dflash_draft(self, asks)
    }
}

/// A rank's runtime as the drafter needs it (the real one is `tp::GlmRank`; the CPU tests drive a recording mock).
pub trait GlmMtpRuntime: LockstepRuntime<Call = GlmCall> {
    fn mtp_book(&self, slot: usize) -> Option<TapBook>;
    fn mtp_draft_keyed(&mut self, asks: &[DraftAsk<'_>], keyed: &[Option<Sampling>]) -> Result<Vec<Option<Drafts>>, ModelError>;
    fn mtp_disable_slot(&mut self, slot: usize);
    /// This rank's window graph cache (lane E: rank 0's backend reports it at `/metrics`). The default: none (the CPU mocks).
    fn graph_counters(&self) -> Option<gb_engine::GraphCounters> {
        None
    }
    /// Whether this rank drafts with DFlash2 (rank 0 of a group with `--draft DIR`, or either rank of a split drafter). The default: no.
    fn has_dflash(&self) -> bool {
        false
    }
    /// Whether this rank holds its half of a split drafter (`GlmModel::dflash_split`): its rounds and prompt ends are forwarded.
    fn dflash_split(&self) -> bool {
        false
    }
    /// A DFlash2 round of a whole drafter on this rank alone (rank 0's: the draft is dense and the head replicated, so no follower takes
    /// part).
    fn dflash_draft(&mut self, _asks: &[DflashAsk<'_>]) -> Result<Vec<Picked>, ModelError> {
        Err(ModelError::Config("this rank has no DFlash2 drafter".into()))
    }
    /// The requests of these asks (`GlmModel::dflash_reqs`): what a split round's call carries.
    fn dflash_reqs(&self, _asks: &[DflashAsk<'_>]) -> Result<Vec<DflashReq>, ModelError> {
        Err(ModelError::Config("this rank has no DFlash2 drafter".into()))
    }
    /// A round of requests (`GlmModel::dflash_run`): a split drafter's half, in step with the peer's same call.
    fn dflash_run(&mut self, _reqs: &[DflashReq]) -> Result<Vec<Picked>, ModelError> {
        Err(ModelError::Config("this rank has no DFlash2 drafter".into()))
    }
    /// The word to the DFlash2 drafter that `slot`'s prompt ends at `end` and takes its next snapshot at `cut`
    /// (`GlmModel::dflash_prompt_end`): host state of a rank that drafts (rank 0's alone with a whole drafter, forwarded with a split one:
    /// [`dflash_prompt_end`]). The default: nothing (a rank without the drafter).
    fn dflash_prompt_end(&mut self, _slot: usize, _end: Option<usize>, _cut: Option<usize>) {}
}

impl<R: GlmMtpRuntime> DraftModel for TpLeader<R> {
    fn has_mtp(&self) -> bool {
        self.inner().has_mtp()
    }
    fn flush(&mut self) -> Result<(), ModelError> {
        TpLeader::flush(self)
    }
    fn mtp_book(&self, slot: usize) -> Option<TapBook> {
        self.inner().mtp_book(slot)
    }
    /// Forwarded first (the follower enters the same NextN forwards and meets this rank in each depth's exchange), then run here; the
    /// follower's digest of its drafts must equal this rank's.
    fn mtp_draft_keyed(&mut self, asks: &[DraftAsk<'_>], keyed: &[Option<Sampling>]) -> Result<Vec<Option<Drafts>>, ModelError> {
        let call = GlmCall::MtpDraft { asks: asks.iter().map(AskWire::of).collect(), keyed: keyed.to_vec() };
        self.forward_call(call, |r, _| r.mtp_draft_keyed(asks, keyed), |d| drafts_digest(d))
    }
    fn has_dflash(&self) -> bool {
        self.inner().has_dflash()
    }
    /// A whole drafter: rank 0 alone, nothing is forwarded (the draft touches only rank 0's own ring and scratch, never the group's
    /// state). A split one: the requests are forwarded first (the follower meets this rank in the round's swaps), then run here; the
    /// follower's digest of its picks must equal this rank's ([`picks_digest`]).
    fn dflash_draft(&mut self, asks: &[DflashAsk<'_>]) -> Result<Vec<Picked>, ModelError> {
        if !self.inner().dflash_split() {
            return self.inner_mut().dflash_draft(asks);
        }
        let reqs = self.inner().dflash_reqs(asks)?;
        self.forward_call(GlmCall::DflashDraft { reqs: reqs.clone() }, |r, _| r.dflash_run(&reqs), |p| picks_digest(p))
    }
}

/// Rank 0's word to the drafter that `slot`'s prompt ends at `end` with its next cut at `cut` (`GlmModel::dflash_prompt_end`): to its own
/// drafter alone when it is whole, forwarded first ([`GlmCall::DflashPromptEnd`]) when it is split (the follower's half absorbs, taps and
/// keeps snapshot windows by the same rule, so both halves of the ring hold the same positions).
pub fn dflash_prompt_end<R: GlmMtpRuntime>(leader: &mut TpLeader<R>, slot: usize, end: Option<usize>, cut: Option<usize>) {
    if !leader.inner().dflash_split() {
        leader.inner_mut().dflash_prompt_end(slot, end, cut);
        return;
    }
    let call = GlmCall::DflashPromptEnd { slot: slot as u32, end: end.map(|e| e as u32), cut: cut.map(|c| c as u32) };
    leader.forward_unit(call, |r| r.dflash_prompt_end(slot, end, cut));
}

/// The boot's agreement check of a split drafter: one round on both ranks (slot 0 at position 0, greedy, seven drafts: it reads no ring
/// position and writes only the drafter's scratch), settled before anything is served, so two ranks whose halves do not make the same picks
/// (another pack, another split, an exchange that does not carry) fail the boot naming `dflash_draft`. Nothing without a split.
pub fn check_split<R: GlmMtpRuntime>(leader: &mut TpLeader<R>) -> Result<(), ModelError> {
    if !leader.inner().dflash_split() {
        return Ok(());
    }
    let req = DflashReq {
        slot: 0,
        p: 0,
        anchor: 0,
        want: crate::draft::DRAFTS,
        coverage: 1.0,
        sampling: Sampling { temperature: 0.0, ..Default::default() },
    };
    let picks = leader.forward_call(GlmCall::DflashDraft { reqs: vec![req] }, |r, _| r.dflash_run(&[req]), |p| picks_digest(p))?;
    leader.settle(leader.last_seq())?;
    eprintln!("rexl3 event=dflash_split_check picks={:?} digest={:016x}", picks[0].tokens, picks_digest(&picks));
    Ok(())
}

/// Rank 0's half of [`GlmCall::MtpDisable`].
pub fn disable_slot<R: GlmMtpRuntime>(leader: &mut TpLeader<R>, slot: usize) {
    leader.forward_unit(GlmCall::MtpDisable { slot: slot as u32 }, |r| r.mtp_disable_slot(slot));
}

/// A follower applies a draft call (`tp::apply_family` routes `MtpDraft` and `MtpDisable` here).
pub fn apply<R: GlmMtpRuntime + ?Sized>(r: &mut R, call: GlmCall) -> Result<CallReply, ModelError> {
    match call {
        GlmCall::MtpDraft { asks, keyed } => {
            let list: Vec<DraftAsk<'_>> = asks.iter().map(AskWire::ask).collect();
            let d = r.mtp_draft_keyed(&list, &keyed)?;
            Ok(CallReply::digest(drafts_digest(&d)))
        }
        GlmCall::MtpDisable { slot } => {
            r.mtp_disable_slot(slot as usize);
            Ok(CallReply::NONE)
        }
        GlmCall::DflashDraft { reqs } => {
            let p = r.dflash_run(&reqs)?;
            Ok(CallReply::digest(picks_digest(&p)))
        }
        GlmCall::DflashPromptEnd { slot, end, cut } => {
            r.dflash_prompt_end(slot as usize, end.map(|e| e as usize), cut.map(|c| c as usize));
            Ok(CallReply::NONE)
        }
        // an image chunk is `tp_vision`'s and a prompt forward's part `tp_part`'s (`tp::apply_family` routes them there): a runtime that
        // reaches here applies neither
        GlmCall::PrefillImage(_) | GlmCall::PrefillPart(_) => {
            Err(ModelError::Request("a prefill call reached the draft calls' dispatch (use tp::apply_family)".into()))
        }
        // the prefix cache's calls key their handles by the call's sequence number (`tp::apply_prefix`): a runtime that reaches here keeps none
        GlmCall::EnableSnapshots { .. }
        | GlmCall::Snapshot { .. }
        | GlmCall::Restore { .. }
        | GlmCall::Release { .. }
        | GlmCall::ArmCut { .. } => {
            Err(ModelError::Request("a prefix-cache call reached the draft calls' dispatch (use tp::apply_prefix)".into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lockstep::{FamilyCall, Msg};

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    fn sample() -> GlmCall {
        GlmCall::MtpDraft {
            asks: vec![AskWire { slot: 1, depth: 3, tokens: vec![7, 8] }, AskWire { slot: 0, depth: 0, tokens: vec![9] }],
            keyed: vec![Some(Sampling { seed: 42, temperature: 0.5, top_k: 20, top_p: 0.95, min_p: 0.0 }), None],
        }
    }

    /// Captured 2026-10-14 (lane L12): length, protocol 4, kind 32, seq 5, then two asks (slot 1, depth 3, tokens [7, 8]; slot 0,
    /// depth 0, tokens [9]) and two samplers (seed 42, temperature 0.5, top_k 20, top_p 0.95, min_p 0; none).
    const GOLDEN_DRAFT: &str = "7000000005200500000000000000020000000000000001000000030000000200000000000000070000000800000000000000000000000100000000000000090000000200000000000000012a00000000000000000000000000e03f1400000000000000666666666666ee3f000000000000000000";
    /// Kind 33, seq 6, slot 3.
    const GOLDEN_DISABLE: &str = "0e0000000521060000000000000003000000";

    #[test]
    fn the_draft_calls_have_these_exact_bytes_and_round_trip() {
        let m: Msg<GlmCall> = Msg::Call(sample());
        let bytes = m.encode(5);
        let (seq, back) = Msg::<GlmCall>::decode(&bytes[4..]).unwrap();
        assert_eq!((seq, &back), (5, &m), "round trip");
        assert_eq!(hex(&bytes), GOLDEN_DRAFT, "the MtpDraft frame's bytes moved");
        let m: Msg<GlmCall> = Msg::Call(GlmCall::MtpDisable { slot: 3 });
        let bytes = m.encode(6);
        let (seq, back) = Msg::<GlmCall>::decode(&bytes[4..]).unwrap();
        assert_eq!((seq, &back), (6, &m));
        assert_eq!(hex(&bytes), GOLDEN_DISABLE, "the MtpDisable frame's bytes moved");
        assert_eq!((sample().kind(), sample().name()), (K_MTP_DRAFT, "mtp_draft"));
        let d = GlmCall::MtpDisable { slot: 0 };
        assert_eq!((d.kind(), d.name()), (K_MTP_DISABLE, "mtp_disable_slot"));
        assert!(!crate::lockstep::CORE_KINDS.contains(&K_MTP_DRAFT) && !crate::lockstep::CORE_KINDS.contains(&K_MTP_DISABLE));
    }

    #[test]
    fn a_truncated_draft_body_is_a_wire_error_and_an_unknown_kind_is_not_the_familys() {
        let bytes = Msg::<GlmCall>::Call(sample()).encode(5);
        for cut in [20usize, 40, bytes.len() - 1] {
            assert!(Msg::<GlmCall>::decode(&bytes[4..cut]).is_err(), "cut at {cut}");
        }
        assert!(GlmCall::decode(36, &mut Dec::new(&[])).unwrap().is_none());
    }

    fn dflash_sample() -> GlmCall {
        let greedy = Sampling { seed: 9, temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0 };
        let keyed = Sampling { seed: 1, temperature: 0.7, top_k: 40, top_p: 0.9, min_p: 0.05 };
        GlmCall::DflashDraft {
            reqs: vec![
                DflashReq { slot: 2, p: 1000, anchor: 77, want: 7, coverage: 0.75, sampling: greedy },
                DflashReq { slot: 0, p: 5, anchor: 3, want: 1, coverage: 1.0, sampling: keyed },
            ],
        }
    }

    /// Kind 34, seq 7: two requests, each slot, position, anchor, drafts wanted and coverage bits (u32s), then its sampler (tag 1, seed,
    /// temperature, `top_k`, `top_p`, `min_p`).
    const GOLDEN_DFLASH: &str = "8c00000005220700000000000000020000000000000002000000e80300004d000000070000000000403f01090000000000000000000000000000000000000000000000000000000000f03f0000000000000000000000000500000003000000010000000000803f010100000000000000666666666666e63f2800000000000000cdccccccccccec3f9a9999999999a93f";
    /// Kind 35, seq 8: slot 1, end 700 (tag 1 and the u32), no cut (tag 0).
    const GOLDEN_PROMPT_END: &str = "14000000052308000000000000000100000001bc02000000";

    /// The split drafter's two calls (kinds 34 and 35): their bytes are pinned, each round-trips, a body one byte short is a wire error,
    /// and the kinds are the family's own, clear of the core's and of every other family call's.
    #[test]
    fn the_split_drafter_calls_have_these_exact_bytes_and_round_trip() {
        let end = GlmCall::DflashPromptEnd { slot: 1, end: Some(700), cut: None };
        for (c, seq, golden, kind, name) in [
            (dflash_sample(), 7u64, GOLDEN_DFLASH, K_DFLASH_DRAFT, "dflash_draft"),
            (end, 8, GOLDEN_PROMPT_END, K_DFLASH_PROMPT_END, "dflash_prompt_end"),
        ] {
            let m: Msg<GlmCall> = Msg::Call(c.clone());
            let bytes = m.encode(seq);
            let (back_seq, back) = Msg::<GlmCall>::decode(&bytes[4..]).unwrap();
            assert_eq!((back_seq, &back), (seq, &m), "round trip of {name}");
            assert_eq!(hex(&bytes), golden, "the {name} frame's bytes moved");
            assert!(Msg::<GlmCall>::decode(&bytes[4..bytes.len() - 1]).is_err(), "{name}");
            assert_eq!((c.kind(), c.name(), c.is_prefix()), (kind, name, false));
            assert!(!crate::lockstep::CORE_KINDS.contains(&kind));
            assert!(![K_MTP_DRAFT, K_MTP_DISABLE, super::super::tp_vision::K_PREFILL_IMAGE, super::super::tp_part::K_PREFILL_PART]
                .contains(&kind));
        }
    }

    /// The split round's ack sees every token, every score bit and every margin bit of every sequence's picks.
    #[test]
    fn the_picks_digest_sees_every_token_score_and_margin() {
        let p = |t: u32, s: f32, m: f32| Picked { tokens: vec![t, 5], score: vec![s, 1.0], margin: vec![m, 2.0], lattice: None };
        let a = vec![p(4, 0.5, 3.0), p(9, -1.0, 0.25)];
        let d = picks_digest(&a);
        assert_eq!(d, picks_digest(&a.clone()), "a pure function");
        for b in [
            vec![p(4, 0.5, 3.0), p(8, -1.0, 0.25)],
            vec![p(4, f32::from_bits(0.5f32.to_bits() + 1), 3.0), p(9, -1.0, 0.25)],
            vec![p(4, 0.5, 3.0), p(9, -1.0, f32::from_bits(0.25f32.to_bits() + 1))],
            vec![p(4, 0.5, 3.0)],
            vec![p(9, -1.0, 0.25), p(4, 0.5, 3.0)],
        ] {
            assert_ne!(picks_digest(&b), d, "{b:?}");
        }
    }

    #[test]
    fn the_digest_sees_every_id_every_value_bit_and_every_switched_off_slot() {
        let a = vec![Some(Drafts { ids: vec![1, 2], vals: vec![0.5, -1.0] }), None];
        let d = drafts_digest(&a);
        assert_eq!(d, drafts_digest(&a.clone()), "a pure function");
        let mut b = a.clone();
        b[0].as_mut().unwrap().ids[1] = 3;
        assert_ne!(drafts_digest(&b), d);
        let mut b = a.clone();
        b[0].as_mut().unwrap().vals[0] = f32::from_bits(0.5f32.to_bits() + 1);
        assert_ne!(drafts_digest(&b), d);
        let mut b = a.clone();
        b[1] = Some(Drafts::default());
        assert_ne!(drafts_digest(&b), d);
        let mut b = a.clone();
        b[0].as_mut().unwrap().vals[1] = -0.0;
        let mut c = a.clone();
        c[0].as_mut().unwrap().vals[1] = 0.0;
        assert_ne!(drafts_digest(&b), drafts_digest(&c), "the value's bits, not its number");
    }

    #[test]
    fn the_vocabulary_fact_names_the_head_each_rank_would_build() {
        assert_eq!(mtp_vocab_fact(false, Some((512, 0.9)), 154_880), "none");
        assert_eq!(mtp_vocab_fact(true, None, 154_880), "full");
        let a = mtp_vocab_fact(true, Some((512, 0.9)), 154_880);
        assert!(a.starts_with("strips=512 share=0.9 ranges="), "{a}");
        assert_ne!(a, mtp_vocab_fact(true, Some((384, 0.9)), 154_880));
        assert_ne!(a, mtp_vocab_fact(true, Some((512, 0.0)), 154_880));
        assert_eq!(a, mtp_vocab_fact(true, Some((512, 0.9)), 154_880));
    }

    // ---------------------------------------------------------------------------------------------- the backend over a recording mock

    use crate::families::glm::spec::SpecConfig;
    use crate::families::glm::tp::GlmFacts;
    use crate::families::glm::tp_backend::tests::{t, Log, Rec};
    use crate::families::glm::tp_backend::GlmTpBackend;
    use crate::lockstep::{run_follower, BootGate, FollowerConfig, FollowerExit, Link};
    use gb_engine::{Backend, ProposeRequest, RowRequest};
    use std::sync::Arc;
    use std::time::Duration;

    type Group = (GlmTpBackend<Rec>, Log, Log, std::thread::JoinHandle<FollowerExit>);

    /// Rank 0's backend with a speculation runtime of `mode`, over a follower thread; the follower drafts with `bias`.
    fn spec_group(mode: gb_spec::SpecMode, bias: u32) -> Group {
        let (a, b) = Link::<GlmCall>::pair(Duration::from_secs(5)).unwrap();
        let (l0, l1): (Log, Log) = (Arc::default(), Arc::default());
        let f = Rec { draft_bias: bias, ..Rec::new(l1.clone()) };
        let h = std::thread::spawn(move || run_follower(f, b, FollowerConfig::new(t())).unwrap());
        let leader = TpLeader::lead(Rec::new(l0.clone()), a, t());
        // these tests pin P1's slicing (4-row chunks); the process default is P2, whose slices land on a 16-row grid
        let be = GlmTpBackend::with_spec(leader, 4, 4, SpecConfig::new(mode))
            .with_contract(crate::families::glm::contract::PrefillContract::P1);
        (be, l0, l1, h)
    }

    /// Two sequences: a prompt each, one window with commits, then a draft round (one keyed, one greedy ask), the window that verifies
    /// it, its commits, and a second round. Returns the drafts of each round.
    fn rounds(be: &mut GlmTpBackend<Rec>) -> Vec<Vec<Option<Vec<u32>>>> {
        let keyed = gb_sampling::Sampling { seed: 77, temperature: 0.7, top_k: 20, top_p: 0.9, min_p: 0.0 };
        let greedy = gb_sampling::Sampling { temperature: 0.0, ..Default::default() };
        let mut seq: Vec<Vec<u32>> = vec![(10..17).collect(), (40..43).collect()];
        let r0 = be.prefill(0, 0, &seq[0], 4);
        let r1 = be.prefill(1, 0, &seq[1], 4);
        assert!(r0.values[0].is_finite() && r1.values[0].is_finite());
        // a window of one row each: the first token after the prompt
        let w = vec![
            RowRequest { slot: 0, tokens: vec![500], first_position: 7 },
            RowRequest { slot: 1, tokens: vec![600], first_position: 3 },
        ];
        be.step(&w, 4);
        be.commit(0, 1);
        be.commit(1, 1);
        seq[0].extend([500, 501]);
        seq[1].extend([600, 601]);
        let mut out = Vec::new();
        for round in 0..2 {
            let reqs = [
                ProposeRequest { slot: 0, tokens: &seq[0], sampling: &keyed, max: 7 },
                ProposeRequest { slot: 1, tokens: &seq[1], sampling: &greedy, max: 7 },
            ];
            let d = be.propose_batch(&reqs);
            // the verify window: the last token and the drafts; keep the first draft of slot 0 and none of slot 1's
            let mut w = Vec::new();
            for (i, s) in seq.iter().enumerate() {
                let mut toks = vec![*s.last().unwrap()];
                toks.extend(d[i].clone().unwrap_or_default());
                w.push(RowRequest { slot: i, tokens: toks, first_position: (s.len() - 1) as u64 });
            }
            be.step(&w, 4);
            let keep0 = if d[0].as_ref().is_some_and(|x| !x.is_empty()) { 2 } else { 1 };
            be.commit(0, keep0);
            be.commit(1, 1);
            for k in 0..keep0 {
                seq[0].push(900 + round * 10 + k as u32);
            }
            seq[1].push(700 + round);
            out.push(d);
        }
        be.release(0);
        be.release(1);
        out
    }

    #[test]
    fn the_follower_replays_rank_zeros_draft_calls_exactly_with_the_depth_and_the_seeds_in_the_call() {
        let (mut be, l0, l1, h) = spec_group(gb_spec::SpecMode::Mtp, 0);
        let d = rounds(&mut be);
        assert!(be.take_error().is_none());
        be.quiesce().unwrap();
        let stats = be.spec.as_ref().unwrap().stats.clone();
        drop(be);
        assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower's call stream is rank 0's");
        let drafts: Vec<&String> = a.iter().filter(|l| l.starts_with("draft")).collect();
        // `--spec mtp` at the default depth 2: slot 0 keyed by its request's seed, slot 1 greedy (its sampler travels; the model leaves a
        // greedy request unkeyed)
        assert_eq!(
            drafts,
            [
                "draft (0 2 [10, 11, 12, 13, 14, 15, 16, 500, 501] Some(77)) (1 2 [40, 41, 42, 600, 601] Some(0))",
                "draft (0 2 [10, 11, 12, 13, 14, 15, 16, 500, 501, 900, 901] Some(77)) (1 2 [40, 41, 42, 600, 601, 700] Some(0))",
            ]
        );
        // every draft call is preceded by the speculation runtime's flush (the single node's `propose` flushes first as well)
        for (i, l) in a.iter().enumerate().filter(|(_, l)| l.starts_with("draft")) {
            assert_eq!(a[i - 1], "flush", "{a:?}");
            let _ = l;
        }
        assert!(d.iter().flatten().all(|x| x.as_ref().is_some_and(|x| x.len() == 2)), "{d:?}");
        // the counters are rank 0's: two rounds of two streams from the head, slot 0 accepted one draft a round
        assert_eq!(stats.rounds[2], 4, "{stats:?}");
        assert_eq!((stats.verified[2], stats.accepted[2]), (8, 2), "{stats:?}");
        assert_eq!(stats.mtp_calls, 2);
    }

    /// Lane L14 (`docs/GLM_TP.md` 11): an IMAGE prompt with drafting on. The image chunks (`PrefillImage`) move the drafter's book on
    /// every rank exactly as a core prefill chunk does (the model's `prefill_chunk_mm` runs `mtp_prefill_chunk`), so the first draft
    /// round after the prompt asks and is answered alike on both ranks; and rank 0's image spans gate the head host-only: a pending
    /// window over an image row forwards no draft call (the same request with no spans would).
    #[test]
    fn an_image_prompt_with_drafting_on_replays_and_the_image_gate_stays_on_rank_zero() {
        use crate::families::glm::model::TapBook;
        use crate::families::glm::tp_backend::tests::{host, rows_fnv, slices, vision};
        use gb_engine::ImageSpan;
        let (be, l0, l1, h) = spec_group(gb_spec::SpecMode::Mtp, 0);
        let mut be = be.with_vision(vision());
        let f = host(4, 1);
        let ims = vec![ImageSpan { start: 1, tokens: 4, merged: (1, 4), features: f.clone(), key: 0 }];
        let mut seq: Vec<u32> = vec![1, 7, 7, 7, 7, 2, 3, 4];
        let r = slices(&mut be, 0, &seq, &ims, 4);
        assert!(r.values.iter().all(|v| v.is_finite()));
        be.step(&[RowRequest { slot: 0, tokens: vec![500], first_position: 8 }], 4);
        be.commit(0, 1);
        seq.extend([500, 501]);
        let greedy = gb_sampling::Sampling { temperature: 0.0, ..Default::default() };
        let d = be.propose_batch(&[ProposeRequest { slot: 0, tokens: &seq, sampling: &greedy, max: 7 }]);
        assert!(d[0].as_ref().is_some_and(|x| x.len() == 2), "the head drafts after an image prompt: {d:?}");
        assert!(be.take_error().is_none());
        // the host-only gate: rank 0's book says rows 3..5 (image rows) are pending; with the slot's spans no draft call is made ...
        be.leader.inner_mut().books[0] = TapBook { ok: true, first: 3, n: 2 };
        let short = &seq[..5];
        let before = l0.lock().unwrap().iter().filter(|l| l.starts_with("draft")).count();
        let g = be.propose_batch(&[ProposeRequest { slot: 0, tokens: short, sampling: &greedy, max: 7 }]);
        assert!(g[0].as_ref().is_none_or(|x| x.is_empty()), "{g:?}");
        assert_eq!(
            l0.lock().unwrap().iter().filter(|l| l.starts_with("draft")).count(),
            before,
            "no draft call over an image row"
        );
        // the predicate the gate asks (`window_is_text`): rows 3..5 overlap the image's 1..5; with no spans they would not
        assert!(
            crate::families::glm::backend::window_is_text(&[], 3, 2)
                && !crate::families::glm::backend::window_is_text(std::slice::from_ref(&(1..5)), 3, 2)
        );
        be.release(0);
        be.quiesce().unwrap();
        drop(be);
        assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower's call stream is rank 0's");
        let head: Vec<&str> = a.iter().take(4).map(|s| s.as_str()).collect();
        assert_eq!(
            head,
            [
                "begin 0",
                &format!("image 0 0 [1, 7, 7, 7] None [(1, 3, \"{}\")]", rows_fnv(&f, 0, 3)),
                &format!("image 0 4 [7, 2, 3, 4] Some(4) [(0, 1, \"{}\")]", rows_fnv(&f, 3, 4)),
                "window [500] [8] [0] 4",
            ]
        );
        let drafts: Vec<&String> = a.iter().filter(|l| l.starts_with("draft")).collect();
        assert_eq!(drafts, ["draft (0 2 [1, 7, 7, 7, 7, 2, 3, 4, 500, 501] Some(0))"]);
    }

    /// Lane L14: the ENGINE (`tp_boot::engine_over_budget` with the flat drafting budget, what `build_leader` serves with `--spec mtp`)
    /// over rank 0's backend with drafting AND images, a request with an image submitted as the server submits it: the image slices
    /// reach the follower as `PrefillImage` calls, the request drafts with the NextN head after its prompt (`MtpDraft` calls), it
    /// finishes, and both ranks saw the same stream.
    #[test]
    fn the_engine_serves_an_image_request_with_drafting_through_the_group() {
        use crate::families::glm::tp_backend::tests::{host, vision};
        use crate::families::glm::tp_boot::engine_over_budget;
        use gb_engine::{ImageSpan, Request};
        use gb_sampling::Sampling;
        use gb_sched::{Admission, ModelMemory};
        let (be, l0, l1, h) = spec_group(gb_spec::SpecMode::Mtp, 0);
        let be = be.with_vision(vision());
        let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 1, activation_reserve: 0 };
        let adm = Admission::new(1 << 30, 0, mm, 4).unwrap();
        let mut e = engine_over_budget(be, adm, true, true);
        let f = host(6, 3);
        let prompt = vec![1, 2, 7, 7, 7, 7, 7, 7, 3, 4];
        let images = vec![ImageSpan { start: 2, tokens: 6, merged: (1, 6), features: f.clone(), key: 0 }];
        let req = Request { prompt, sampling: Sampling::default(), max_new: 12, stop: vec![], draft: true, images, cache: false };
        let id = e.submit(req).unwrap();
        let mut done = false;
        for _ in 0..400 {
            e.step().unwrap();
            if e.take_events().iter().any(|ev| ev.id == id && ev.finish.is_some()) {
                done = true;
                break;
            }
        }
        assert!(done, "the image request did not finish");
        drop(e);
        h.join().unwrap();
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "the follower's call stream is rank 0's");
        assert!(a.iter().any(|s| s.starts_with("image 0 0 [1, 2, 7, 7]")), "{a:?}");
        let first_draft = a.iter().position(|s| s.starts_with("draft")).unwrap_or_else(|| panic!("no draft call: {a:?}"));
        let last_image = a.iter().rposition(|s| s.starts_with("image")).unwrap();
        assert!(first_draft > last_image, "the head drafts after the image prompt: {a:?}");
    }

    #[test]
    fn a_follower_that_drafts_otherwise_loses_the_group_naming_the_draft_call() {
        let (mut be, _l0, _l1, h) = spec_group(gb_spec::SpecMode::Mtp, 1);
        let _ = rounds(&mut be);
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while be.leader.failure().is_none() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let why = be.leader.failure().expect("the group is lost");
        assert!(why.contains("mtp_draft"), "{why}");
        // the next fallible call says so, fatally
        let r = be.prefill(2, 0, &[1, 2], 2);
        assert!(r.values.iter().all(|v| v.is_nan()));
        let e = be.take_error().expect("an error");
        assert!(e.fatal, "{e:?}");
        drop(be);
        let _ = h.join();
    }

    #[test]
    fn a_disabled_slot_is_disabled_on_every_rank_and_drafts_nothing() {
        let (mut be, l0, l1, h) = spec_group(gb_spec::SpecMode::Mtp, 0);
        let p: Vec<u32> = (10..17).collect();
        be.prefill(0, 0, &p, 4);
        disable_slot(&mut be.leader, 0);
        let s = gb_sampling::Sampling::default();
        let mut seq = p.clone();
        seq.push(1);
        let d = be.propose_batch(&[ProposeRequest { slot: 0, tokens: &seq, sampling: &s, max: 7 }]);
        assert_eq!(d, vec![Some(vec![])], "no head drafts for a disabled slot");
        be.quiesce().unwrap();
        drop(be);
        h.join().unwrap();
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b);
        assert!(a.contains(&"disable 0".to_string()) && !a.iter().any(|l| l.starts_with("draft")), "{a:?}");
    }

    #[test]
    fn spec_none_forwards_no_draft_call_and_auto_forwards_only_what_the_arbiter_chose() {
        let (mut be, l0, _l1, h) = spec_group(gb_spec::SpecMode::None, 0);
        let _ = rounds(&mut be);
        drop(be);
        h.join().unwrap();
        assert!(!l0.lock().unwrap().iter().any(|l| l.starts_with("draft") || l == "flush"));
        let (mut be, l0, l1, h) = spec_group(gb_spec::SpecMode::Auto, 0);
        let _ = rounds(&mut be);
        assert!(be.take_error().is_none());
        be.quiesce().unwrap();
        drop(be);
        h.join().unwrap();
        let (a, b) = (l0.lock().unwrap().clone(), l1.lock().unwrap().clone());
        assert_eq!(a, b, "whatever rank 0's arbiter chose, the follower replays it");
    }

    // ------------------------------------------------------------------------------------------------------------ the boot gate

    fn mtp_facts() -> GlmFacts {
        GlmFacts {
            graphs: false,
            overlap: 0,
            graph_at_first: false,
            draft_graphs: false,
            prefill_small: 0,
            mtp: true,
            mtp_depth: 4,
            window_rows: 64,
            mtp_vocab: mtp_vocab_fact(true, Some((512, 0.9)), 154_880),
            ownership: 0x0123_4567_89ab_cdef,
            collective: "ring".into(),
            vision: false,
            split: 0,
            prefill: "contract=p1".into(),
            cache_request: "off".into(),
            cache_slots: 0,
            cache_slot_bytes: 0,
            cache_pages: 0,
            kv: "fp16".into(),
            taps: "none".into(),
            draft_split: "off".into(),
        }
    }

    #[test]
    fn the_boot_gate_refuses_a_rank_without_the_layer_another_depth_cap_or_another_head_by_name() {
        use crate::families::glm::model::ModelOptions;
        use crate::families::glm::tp::gate;
        let mo = ModelOptions { slots: 2, kv_blocks: 64, max_context: 8192, layers: 45 };
        let f = mtp_facts();
        let g0 = gate("p".into(), &mo, 512, f.clone()).at(0, 2);
        assert!(BootGate::check(&g0, &gate("p".into(), &mo, 512, f.clone()).at(1, 2)).is_ok());
        let mut off = GlmFacts { mtp: false, mtp_depth: 0, mtp_vocab: "none".into(), ..f.clone() };
        let e = BootGate::check(&g0, &gate("p".into(), &mo, 512, off.clone()).at(1, 2)).unwrap_err();
        assert!(e.contains("the NextN draft layer (--spec): rank 0 has on, rank 1 has off"), "{e}");
        assert!(e.contains("the draft head's vocabulary (REXL3_MTP_VOCAB)"), "{e}");
        off.mtp = true;
        off.mtp_depth = 3;
        off.mtp_vocab = f.mtp_vocab.clone();
        let e = BootGate::check(&g0, &gate("p".into(), &mo, 512, off).at(1, 2)).unwrap_err();
        assert!(e.contains("the draft depth cap (MTP_MAX_DEPTH): rank 0 has 4, rank 1 has 3"), "{e}");
        for v in
            ["full".to_string(), mtp_vocab_fact(true, Some((512, 0.0)), 154_880), mtp_vocab_fact(true, Some((384, 0.9)), 154_880)]
        {
            let other = GlmFacts { mtp_vocab: v.clone(), ..f.clone() };
            let e = BootGate::check(&g0, &gate("p".into(), &mo, 512, other).at(1, 2)).unwrap_err();
            assert!(e.contains("the draft head's vocabulary (REXL3_MTP_VOCAB)") && e.contains(&v), "{e}");
        }
    }

    /// Captured 2026-10-14 (lane L12): the GLM Hello with the NextN layer: as the golden of `tp_backend` (mtp off) but mtp 1, then the
    /// depth cap 4 (u32), then the vocabulary fact `strips=512 share=0.9 ranges=...`. Lane L14 (the merge with lane L13) moved it by
    /// exactly the vision fact L13 appended after the collective: one byte `00` after `ring`, length 0x9f -> 0xa0 (L12's capture had no
    /// vision fact; every other byte is L12's). Lane P (2026-10-18) appended the prefill contract fact after the vision byte
    /// (`contract=p1`, u64 length 11 then its bytes): length 0xa0 -> 0xb3.
    const GOLDEN_HELLO_MTP: &str = "b300000005010000000000000000525854500100000000000000702d000000020000001000000000000000001000000000000000020000000104000000400000002c000000000000007374726970733d3531322073686172653d302e392072616e6765733d39303636623965633730333536613961efcdab8967452301040000000000000072696e67000b00000000000000636f6e74726163743d7031010000000000000062010000000200000001000000000000006e";

    /// Lane L14: the Hello of rank 0 with the NextN layer AND the vision tower (the combined configuration, `docs/GLM_TP.md` 11): the
    /// MTP facts in their place, then the leader-only vision fact (`01` after `ring`), then Lane P's prefill contract fact. The same
    /// length as [`GOLDEN_HELLO_MTP`].
    const GOLDEN_HELLO_MTP_VISION: &str = "b300000005010000000000000000525854500100000000000000702d000000020000001000000000000000001000000000000000020000000104000000400000002c000000000000007374726970733d3531322073686172653d302e392072616e6765733d39303636623965633730333536613961efcdab8967452301040000000000000072696e67010b00000000000000636f6e74726163743d7031010000000000000062010000000200000001000000000000006e";

    #[test]
    fn the_hello_with_the_nextn_layer_carries_the_depth_cap_and_the_head_with_these_exact_bytes() {
        use crate::families::glm::model::ModelOptions;
        use crate::families::glm::tp::gate;
        let mo = ModelOptions { slots: 2, kv_blocks: 16, max_context: 4096, layers: 45 };
        let mut g = gate("p".into(), &mo, 512, mtp_facts()).at(1, 2);
        g.build = "b".into();
        g.nccl = "n".into();
        let m: Msg<GlmCall> = Msg::Hello(g);
        let bytes = m.encode(0);
        let (_, back) = Msg::<GlmCall>::decode(&bytes[4..]).unwrap();
        assert_eq!(back, m, "round trip");
        assert_eq!(hex(&bytes), GOLDEN_HELLO_MTP, "the GLM Hello's bytes (NextN on) moved");
        // the facts of a rank without the layer keep the bytes they had (tp_backend's golden): no depth on the wire
        let mut e = Enc::new();
        <GlmCall as FamilyCall>::encode_facts(&GlmFacts::local(288, 2), &mut e);
        let mut d = Dec::new(&e.0);
        assert_eq!(<GlmCall as FamilyCall>::decode_facts(&mut d).unwrap(), GlmFacts::local(288, 2));
    }

    /// The combined configuration (lane L14): rank 0 with the NextN layer and the tower. The order is MTP facts in their place, the
    /// leader-only vision fact last; a follower with the NextN layer and no tower meets it (vision is not compared as an equality), a
    /// follower with a tower is refused by name whatever the leader holds, and an MTP mismatch is still refused beside it.
    #[test]
    fn the_hello_with_the_nextn_layer_and_the_tower_has_the_vision_fact_last_with_these_exact_bytes() {
        use crate::families::glm::model::ModelOptions;
        use crate::families::glm::tp::gate;
        use crate::lockstep::BootGate;
        let mo = ModelOptions { slots: 2, kv_blocks: 16, max_context: 4096, layers: 45 };
        // rank 0's Hello (pinned build and NCCL fingerprint, placed as rank 0)
        let mut g = gate("p".into(), &mo, 512, mtp_facts().with_vision(true)).at(1, 2);
        g.build = "b".into();
        g.nccl = "n".into();
        let m: Msg<GlmCall> = Msg::Hello(g);
        let bytes = m.encode(0);
        let (_, back) = Msg::<GlmCall>::decode(&bytes[4..]).unwrap();
        assert_eq!(back, m, "round trip");
        assert_eq!(hex(&bytes), GOLDEN_HELLO_MTP_VISION, "the GLM Hello's bytes (NextN on, tower on) moved");
        // the two differ in the vision byte only
        let off = GOLDEN_HELLO_MTP.as_bytes();
        let on = GOLDEN_HELLO_MTP_VISION.as_bytes();
        let differ: Vec<usize> = (0..off.len()).filter(|&i| off[i] != on[i]).collect();
        assert_eq!(differ.len(), 1, "{differ:?}");
        // the gate: leader (NextN + tower) over a follower (NextN, no tower) agrees
        let leader = gate("p".into(), &mo, 512, mtp_facts().with_vision(true)).at(0, 2);
        assert!(BootGate::check(&leader, &gate("p".into(), &mo, 512, mtp_facts()).at(1, 2)).is_ok());
        // a follower with a tower: refused by name, with or without the leader's
        for l in [&leader, &gate("p".into(), &mo, 512, mtp_facts()).at(0, 2)] {
            let e = BootGate::check(l, &gate("p".into(), &mo, 512, mtp_facts().with_vision(true)).at(1, 2)).unwrap_err();
            assert!(e.contains("the vision tower (--no-vision)") && e.contains("rank 1 has one"), "{e}");
        }
        // a follower without the NextN layer: refused by name, the vision fact not in the way
        let e =
            BootGate::check(&leader, &gate("p".into(), &mo, 512, GlmFacts { vision: false, ..GlmFacts::local(288, 2) }).at(1, 2))
                .unwrap_err();
        assert!(e.contains("the NextN draft layer (--spec): rank 0 has on, rank 1 has off") && !e.contains("vision"), "{e}");
    }
}
