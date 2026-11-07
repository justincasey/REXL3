//! Prompt forwards of several prompts, and in parts, on a GLM tensor-parallel group (`model/part.rs`): the family call [`PrefillPart`]
//! (kind 41). Rank 0's backend decides; every rank runs the same `GlmModel::prefill_rows` / `prefill_resume` with the call's arguments, so
//! the ranks' forwards and their pauses are the same calls in the same order (a paused forward is the rank's own state, resumed by the next
//! part call; what runs meanwhile is the same windows and forwards on every rank). Acked with the candidates' digest, as a core prefill.

use super::tp::{GlmCall, GlmRank};
use crate::error::ModelError;
use crate::forward::Candidates;
use crate::lockstep::{candidates_digest, dec_batch, enc_batch, CallReply, Dec, Enc, LockstepRuntime, TpLeader, WireError};
use crate::rows::RowBatch;
use gb_engine::RowRequest;

/// The frame kind of [`PrefillPart`].
pub const K_PREFILL_PART: u8 = 41;

/// The layer bound of a part that runs the forward to its end.
pub const TO_END: u32 = u32::MAX;

/// Rows a forward may have while a prompt forward is paused (the stash every P2 rank holds, `GlmModel::enable_prefill_parts`): fresh
/// short prompts (and windows) run meanwhile up to these rows.
pub const PART_ROOM: usize = 512;

/// `REXL3_GLM_PREFILL_PART_MS`: while a sequence decodes, rank 0 runs a long prompt slice in parts of about this many milliseconds (at
/// least a layer), a decode round between two (default 100; `0`/`off`: every slice whole). Rank 0's alone (its parts are calls every rank
/// runs); never a bit. Smaller parts give the decoding streams more of the device and a long prompt less (MEASURED on pair B, the bench's
/// mixed and trace phases: 300 ms -> trace mean 40.5 s, a 32K prompt beside 4 chats 39.5 s; 150 ms -> 33.1 s, 50.0 s; 100 ms -> 27.9 s,
/// 53.6 s; 80 ms -> 28.5 s, 55.0 s).
pub const PART_MS_ENV: &str = "REXL3_GLM_PREFILL_PART_MS";

/// `REXL3_GLM_PREFILL_BATCH_ROWS`: while a sequence decodes, rank 0 puts at most this many rows of fresh short prompts (each at most
/// [`PART_ROOM`] rows) through one forward, which the decoding streams wait for whole (default 2,048; with nothing decoding, the chunk's
/// rows). Rank 0's alone; never a bit.
pub const BATCH_ROWS_ENV: &str = "REXL3_GLM_PREFILL_BATCH_ROWS";

/// The batch rows of [`BATCH_ROWS_ENV`], refused by name when malformed.
pub fn batch_rows_from_env() -> Result<usize, ModelError> {
    match std::env::var(BATCH_ROWS_ENV).ok().as_deref().map(str::trim) {
        None | Some("") => Ok(2048),
        Some(s) => s.parse::<usize>().map_err(|_| ModelError::Config(format!("{BATCH_ROWS_ENV}={s:?}: expected rows"))),
    }
}

/// The part target of [`PART_MS_ENV`] (0: parts off), refused by name when malformed.
pub fn part_ms_from_env() -> Result<f64, ModelError> {
    parse_part_ms(std::env::var(PART_MS_ENV).ok().as_deref())
}

fn parse_part_ms(v: Option<&str>) -> Result<f64, ModelError> {
    match v.map(str::trim) {
        None | Some("") => Ok(100.0),
        Some("off") => Ok(0.0),
        Some(s) => s
            .parse::<f64>()
            .ok()
            .filter(|x| x.is_finite() && *x >= 0.0)
            .ok_or_else(|| ModelError::Config(format!("{PART_MS_ENV}={s:?}: expected milliseconds (0 or `off`: whole slices)"))),
    }
}

/// A rank's stash of [`PART_ROOM`] rows, under the P2 contract (the only one whose forwards run several prompts or in parts). Called
/// after the drafter is attached (the stash keeps its tap rows).
pub fn enable(m: &mut super::model::GlmModel) -> Result<(), ModelError> {
    if m.prefill_contract() != super::contract::PrefillContract::P2 {
        return Ok(());
    }
    let bytes = m.enable_prefill_parts(PART_ROOM)?;
    eprintln!("rexl3 event=glm_prefill_parts room_rows={} stash_mib={:.1}", m.prefill_part_rows(), bytes as f64 / (1u64 << 20) as f64);
    Ok(())
}

/// One part of a prompt forward: `begin` the forward's rows (each sequence's consecutive prompt rows, as `RowBatch::from_requests` makes
/// them) when it starts, `None` when the paused one goes on; its layers up to `until` (exclusive; [`TO_END`]: the forward's end, with the
/// candidates of each sequence's last row when `want`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefillPart {
    pub begin: Option<RowBatch>,
    pub want: Option<u32>,
    pub until: u32,
}

impl PrefillPart {
    pub fn encode(&self, e: &mut Enc) {
        e.bool(self.begin.is_some());
        if let Some(b) = &self.begin {
            enc_batch(e, b);
        }
        e.opt_u32(self.want);
        e.u32(self.until);
    }

    pub fn decode(d: &mut Dec<'_>) -> Result<PrefillPart, WireError> {
        let begin = if d.bool("prefill_part begins")? { Some(dec_batch(d)?) } else { None };
        Ok(PrefillPart { begin, want: d.opt_u32("prefill_part want")?, until: d.u32("prefill_part until")? })
    }

    /// The forward's rows, refused (before anything is launched, so on every rank alike) unless they are exactly what
    /// `RowBatch::from_requests` makes of their sequences: each sequence's consecutive positions, in its own slot below `slots`.
    pub fn check(&self, slots: usize) -> Result<(), ModelError> {
        let Some(b) = &self.begin else { return Ok(()) };
        let bad = || ModelError::Request("a prompt forward is the consecutive rows of its sequences, each in a slot of the model".into());
        if b.n_seq() == 0 || b.rows == 0 || b.ids.len() != b.rows || b.seq_start.len() != b.n_seq() + 1 {
            return Err(bad());
        }
        let mut reqs = Vec::with_capacity(b.n_seq());
        for i in 0..b.n_seq() {
            let (slot, r) = (b.seq_slot[i], b.seq_rows(i));
            if slot < 0 || slot as usize >= slots || r.is_empty() || r.end > b.rows || b.positions[r.start] < 0 {
                return Err(bad());
            }
            let tokens = b.ids[r.clone()].iter().map(|&t| u32::try_from(t).map_err(|_| bad())).collect::<Result<Vec<u32>, _>>()?;
            reqs.push(RowRequest { slot: slot as usize, tokens, first_position: b.positions[r.start] as u64 });
        }
        if RowBatch::from_requests(&reqs) != *b {
            return Err(bad());
        }
        Ok(())
    }
}

/// A runtime that can run [`PrefillPart`]s (every rank of a group that serves them). The defaults are a runtime that cannot: rank 0's
/// backend then prefills one prompt a call, whole.
pub trait PartRuntime: LockstepRuntime<Call = GlmCall> {
    /// Whether forwards of several prompts run (the P2 contract: `model/part.rs`).
    fn prefill_batches(&self) -> bool {
        false
    }
    /// The rows a forward may have while another is paused (0: no forward runs in parts).
    fn prefill_part_rows(&self) -> usize {
        0
    }
    /// The layers a forward runs (a part's bound is a layer).
    fn prefill_layers(&self) -> usize {
        0
    }
    fn prefill_part(&mut self, _call: &PrefillPart) -> Result<Option<Candidates>, ModelError> {
        Err(ModelError::NotBuilt("GLM prompt forwards of several prompts or in parts".into()))
    }
}

impl PartRuntime for GlmRank {
    fn prefill_batches(&self) -> bool {
        self.m.prefill_contract() == super::contract::PrefillContract::P2
    }

    fn prefill_part_rows(&self) -> usize {
        self.m.prefill_part_rows()
    }

    fn prefill_layers(&self) -> usize {
        self.m.layers()
    }

    fn prefill_part(&mut self, call: &PrefillPart) -> Result<Option<Candidates>, ModelError> {
        call.check(self.m.slots)?;
        let want = call.want.map(|w| w as usize);
        let until = if call.until == TO_END { usize::MAX } else { call.until as usize };
        match &call.begin {
            Some(b) => {
                let one = (b.n_seq() == 1).then(|| b.seq_slot[0] as usize);
                let t = self.m.slices.enter(super::profile_slices::Kind::Prefill, b.rows, one);
                let r = self.m.prefill_rows(b, want, until);
                self.m.slices.leave(t, r.is_ok());
                r
            }
            None => self.m.prefill_resume(until),
        }
    }
}

/// Rank 0: forwards `call` and applies it here (the same decoded bytes on every rank); the ack's digest is the candidates'.
pub fn forward_part<R: PartRuntime>(leader: &mut TpLeader<R>, call: PrefillPart) -> Result<Option<Candidates>, ModelError> {
    let local = call.clone();
    leader.forward_call(
        GlmCall::PrefillPart(call),
        move |m, _| m.prefill_part(&local),
        |c: &Option<Candidates>| c.as_ref().map_or(0, candidates_digest),
    )
}

/// The follower's answer to a [`PrefillPart`]: the candidates' digest, compared with rank 0's.
pub fn apply_part<R: PartRuntime>(rt: &mut R, call: &PrefillPart) -> Result<CallReply, ModelError> {
    rt.prefill_part(call).map(|c| CallReply::digest(c.as_ref().map_or(0, candidates_digest)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(seqs: &[(usize, u64, &[u32])]) -> RowBatch {
        RowBatch::from_requests(
            &seqs.iter().map(|&(slot, at, t)| RowRequest { slot, tokens: t.to_vec(), first_position: at }).collect::<Vec<_>>(),
        )
    }

    #[test]
    fn a_part_round_trips_the_wire_and_its_rows_are_checked() {
        for c in [
            PrefillPart { begin: Some(batch(&[(1, 0, &[5, 6, 7]), (3, 16, &[9])])), want: Some(4), until: TO_END },
            PrefillPart { begin: Some(batch(&[(0, 0, &[1; 20])])), want: None, until: 12 },
            PrefillPart { begin: None, want: Some(2), until: 30 },
        ] {
            let mut e = Enc::default();
            c.encode(&mut e);
            let mut d = Dec::new(&e.0);
            assert_eq!(PrefillPart::decode(&mut d).unwrap(), c);
            d.done("prefill_part").unwrap();
            c.check(4).unwrap();
        }
        // a slot past the model's, and rows that are not one request's consecutive positions, are refused before anything runs
        let past = PrefillPart { begin: Some(batch(&[(4, 0, &[5])])), want: None, until: TO_END };
        assert!(past.check(4).is_err());
        let mut gap = batch(&[(1, 0, &[5, 6, 7])]);
        gap.positions[2] = 9;
        assert!(PrefillPart { begin: Some(gap), want: None, until: TO_END }.check(4).is_err());
    }
}
