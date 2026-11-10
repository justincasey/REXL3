use super::*;
use crate::error::ModelError;
use crate::forward::{Candidates, PrefillPath};
use crate::rows::RowBatch;
use gb_engine::{Backend, ProposeRequest, RowRequest};

#[derive(Default)]
struct Fake {
    calls: Vec<(usize, usize, PrefillPath, Option<usize>)>,
    mtp: Vec<(usize, Vec<u32>, Option<u32>, PrefillPath)>,
    fail: Option<ModelError>,
    drafts: Vec<usize>,
    committed: Vec<usize>,
    /// `window_candidates` answers with a clone of this table when set (else the all-7 default).
    next_cands: Option<Candidates>,
    /// When true, `window_candidates_dv` packs column 0 of `next_cands` (the kernel's gather): the
    /// runtime supports the device gather, so it is always engaged (there is no switch).
    dv: bool,
}
impl FlashNextRuntime for Fake {
    fn vocab_size(&self) -> usize {
        42
    }
    fn has_mtp(&self) -> bool {
        true
    }
    fn begin_slot(&mut self, _: usize) -> Result<(), ModelError> {
        Ok(())
    }
    fn end_slot(&mut self, _: usize) {}
    fn prefill_chunk(&mut self, b: &RowBatch, path: PrefillPath, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        self.calls.push((b.positions[0] as usize, b.rows, path, want));
        if let Some(e) = self.fail.take() {
            return Err(e);
        }
        Ok(want.map(|count| Candidates { rows: 1, count, ids: vec![7; count], vals: vec![1.0; count] }))
    }
    fn mtp_after_prefill(
        &mut self,
        _: usize,
        first: usize,
        chunk: &[u32],
        next: Option<u32>,
        path: PrefillPath,
    ) -> Result<(), ModelError> {
        self.mtp.push((first, chunk.to_vec(), next, path));
        Ok(())
    }
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        if let Some(e) = self.fail.take() {
            return Err(e);
        }
        Ok(self.next_cands.clone().unwrap_or_else(|| Candidates {
            rows: b.rows,
            count,
            ids: vec![7; b.rows * count],
            vals: vec![1.0; b.rows * count],
        }))
    }
    fn window_candidates_dv(&mut self, b: &RowBatch, count: usize) -> Result<Option<Candidates>, ModelError> {
        if !self.dv {
            return Ok(None);
        }
        if let Some(e) = self.fail.take() {
            return Err(e);
        }
        let c = self.next_cands.clone().unwrap_or_else(|| Candidates {
            rows: b.rows,
            count,
            ids: vec![7; b.rows * count],
            vals: vec![1.0; b.rows * count],
        });
        // the host mirror of the gather kernel: pack column 0, NaN rows propagate
        let (ids, vals) = gb_cuda::qfn_verify::greedy_gather_host(&c.ids, &c.vals, c.count);
        Ok(Some(Candidates { rows: c.rows, count: 1, ids, vals }))
    }
    fn commit(&mut self, _: usize, keep: usize) {
        self.committed.push(keep);
    }
    fn flush(&mut self) -> Result<(), ModelError> {
        Ok(())
    }
    fn mtp_propose(&mut self, _: usize, _: &[u32], k: usize) -> Result<(Vec<u32>, Vec<f32>), ModelError> {
        self.drafts.push(k);
        Ok((vec![7; k], vec![0.5; k]))
    }
}

#[test]
fn prompt_total_fixes_path_for_one_row_tail_and_resume_grid() {
    let mut b = FlashNextBackend::with_runtime(Fake::default(), 64, 6, true);
    let prompt = vec![3; 130];
    for done in [0, 64, 128] {
        let step = b.prefill_slice(1, &prompt, &[], done, 1);
        assert_eq!(step.done, (done + 64).min(130));
    }
    assert_eq!(
        b.m.calls,
        vec![(0, 64, PrefillPath::Tiled, None), (64, 64, PrefillPath::Tiled, None), (128, 2, PrefillPath::Tiled, Some(1))]
    );
    assert_eq!(b.m.mtp[0].2, Some(3));
    let mut short = FlashNextBackend::with_runtime(Fake::default(), 64, 0, true);
    short.prefill_slice(1, &vec![3; 128], &[], 0, 1);
    assert_eq!(short.m.calls[0].2, PrefillPath::Gemv);
    assert_eq!(next_chunk_end(67, 130, 64), 128);
    assert_eq!(next_chunk_end(128, 129, 64), 129);
}

#[test]
fn piece_keeps_total_path_even_if_first_piece_is_short() {
    let mut b = FlashNextBackend::with_runtime(Fake::default(), 64, 0, true);
    b.prefill_piece(1, 0, &vec![3; 65], 129, 1);
    assert_eq!(b.m.calls, vec![(0, 64, PrefillPath::Tiled, None), (64, 1, PrefillPath::Tiled, Some(1))]);
    assert_eq!(b.m.mtp.len(), 0);
}

#[test]
fn mtp_depth_grows_with_acceptance_and_is_bounded_by_request() {
    let mut b = FlashNextBackend::with_runtime(Fake::default(), 64, 6, true);
    let sampling = gb_sampling::Sampling::default();
    let req = [ProposeRequest { slot: 1, tokens: &[3], sampling: &sampling, max: 4 }];
    assert_eq!(b.propose_batch_scored(&req)[0].as_ref().unwrap().tokens.len(), 4);
    b.commit(1, 2);
    b.propose_batch_scored(&req);
    b.commit(1, 5);
    b.propose_batch_scored(&req);
    assert_eq!(b.m.drafts, vec![4, 3, 4]);
    b.release(1);
    b.propose_batch_scored(&req);
    assert_eq!(b.m.drafts[3], 4);
    let multi = [
        ProposeRequest { slot: 1, tokens: &[3], sampling: &sampling, max: 6 },
        ProposeRequest { slot: 2, tokens: &[3], sampling: &sampling, max: 6 },
    ];
    assert_eq!(b.propose_batch(&multi), vec![None, None]);
    assert_eq!(b.m.drafts.len(), 4);
}

#[test]
fn overflowing_prefill_start_records_a_request_error() {
    let mut b = FlashNextBackend::with_runtime(Fake::default(), 64, 0, true);
    let row = b.prefill(1, u64::MAX, &[3], 1);
    assert!(row.values[0].is_nan());
    assert!(!b.take_error().unwrap().fatal);
    assert!(b.m.calls.is_empty());
}

#[test]
fn errors_are_reported_once_and_fatal_errors_latch() {
    let mut b = FlashNextBackend::with_runtime(Fake::default(), 64, 0, true);
    b.m.fail = Some(ModelError::Request("bad input".into()));
    assert!(b.step(&[RowRequest { slot: 0, first_position: 0, tokens: vec![3] }], 1)[0][0].values[0].is_nan());
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("bad input"));
    assert!(b.take_error().is_none());
    b.m.fail = Some(ModelError::StateLost("lost state".into()));
    b.step(&[RowRequest { slot: 0, first_position: 0, tokens: vec![3] }], 1);
    let e = b.take_error().unwrap();
    assert!(e.fatal && e.message.contains("lost state"));
    b.step(&[RowRequest { slot: 0, first_position: 0, tokens: vec![3] }], 1);
    assert!(b.take_error().unwrap().fatal);
    assert!(b.m.fail.is_none());
}

/// Drives `prompt` through `prefill_slice` with the engine's row limit `limit_at(done)` set before each slice, as `Engine::step` does
/// when a deadline comes near and goes: the slice ends and the last row.
fn drive_limited<M: FlashNextRuntime>(
    b: &mut FlashNextBackend<M>,
    prompt: &[u32],
    limit_at: impl Fn(usize) -> Option<usize>,
) -> (Vec<usize>, Option<gb_engine::RowOutput>) {
    let (mut ends, mut done) = (Vec::new(), 0);
    loop {
        assert!(b.set_prefill_row_limit(limit_at(done)), "Flash-Next supports the row limit");
        let step = b.prefill_slice(1, prompt, &[], done, 4);
        b.set_prefill_row_limit(None);
        assert!(step.done > done, "no progress at {done}");
        done = step.done;
        ends.push(done);
        if step.row.is_some() || done >= prompt.len() {
            return (ends, step.row);
        }
    }
}

#[test]
fn the_row_limit_puts_slices_on_its_absolute_grid_and_keeps_the_prompt_path_and_mtp_feed() {
    let prompt: Vec<u32> = (0..300).map(|i| (i % 41) as u32).collect();
    let mut plain = FlashNextBackend::with_runtime(Fake::default(), 4096, 6, true);
    let (ends0, row0) = drive_limited(&mut plain, &prompt, |_| None);
    assert_eq!(ends0, vec![300]);
    let mut lim = FlashNextBackend::with_runtime(Fake::default(), 4096, 6, true);
    let (ends, row) = drive_limited(&mut lim, &prompt, |_| Some(128));
    assert_eq!(ends, vec![128, 256, 300]);
    assert_eq!(
        lim.m.calls,
        vec![(0, 128, PrefillPath::Tiled, None), (128, 128, PrefillPath::Tiled, None), (256, 44, PrefillPath::Tiled, Some(4))]
    );
    assert!(row.unwrap().bits_eq(&row0.unwrap()));
    // the MTP head is fed every token once, each with the token after it, on the prompt's path
    let fed: Vec<u32> = lim.m.mtp.iter().flat_map(|(_, t, _, _)| t.clone()).collect();
    assert_eq!(fed, prompt);
    let nexts: Vec<Option<u32>> = lim.m.mtp.iter().map(|m| m.2).collect();
    assert_eq!(nexts, vec![Some(prompt[128]), Some(prompt[256]), None]);
    assert!(lim.m.mtp.iter().all(|m| m.3 == PrefillPath::Tiled));
    // the limit comes on mid-prompt and goes again: the slices stay on the absolute grids
    let long = vec![5u32; 700];
    let mut b = FlashNextBackend::with_runtime(Fake::default(), 256, 0, true);
    let (ends, _) = drive_limited(&mut b, &long, |done| (256..512).contains(&done).then_some(128));
    assert_eq!(ends, vec![256, 384, 512, 700]);
    // a zero limit is one row, not a panic; a limit above the chunk changes nothing
    let mut b = FlashNextBackend::with_runtime(Fake::default(), 64, 0, true);
    assert_eq!(drive_limited(&mut b, &[5, 5, 5], |_| Some(0)).0, vec![1, 2, 3]);
    let mut b = FlashNextBackend::with_runtime(Fake::default(), 64, 0, true);
    assert_eq!(drive_limited(&mut b, &vec![5u32; 130], |_| Some(128)).0, vec![64, 128, 130]);
    assert!(b.take_error().is_none());
}

/// Run only on an idle machine: FlashNextModel::load owns the full GPU pack and
/// the warm n-gram table. This bitwise off-grid check remains GPU-gated.
#[test]
#[ignore]
fn real_pack_off_grid_piece_equals_whole_bits() {
    let Some(dir) = std::env::var_os("REXL3_FLASH_NEXT_DIR") else {
        eprintln!("not run: REXL3_FLASH_NEXT_DIR needs an idle GPU box");
        return;
    };
    let home = std::env::var_os("HOME").unwrap();
    let oracle = std::env::var_os("REXL3_FLASH_NEXT_ORACLE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(home).join("rexl3-oracle/ref305"));
    let raw = std::fs::read(oracle.join("ids_0.u32")).unwrap();
    let tokens: Vec<u32> = raw.chunks_exact(4).take(1024).map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect();
    assert_eq!(tokens.len(), 1024);
    let m = super::super::model::FlashNextModel::load(
        std::path::Path::new(&dir),
        super::super::model::ModelOptions { slots: 1, kv_pages: 12, max_context: 2048, prefill_rows: 256, warm: false },
    )
    .unwrap();
    let mut b = FlashNextBackend::new(m, 256, 0);
    let whole = b.prefill_piece(0, 0, &tokens, tokens.len(), 4);
    assert!(b.take_error().is_none());
    b.release(0);
    let _ = b.prefill_piece(0, 0, &tokens[..513], tokens.len(), 4);
    let split = b.prefill_piece(0, 513, &tokens[513..], tokens.len(), 4);
    assert!(b.take_error().is_none());
    assert!(whole.bits_eq(&split), "1024-token prompt: off-grid 513 split changed candidate bits");
}

/// One candidate table: 4 rows, 3 candidates each, sorted by (value DESC, id ASC) so column 0 is
/// the greedy winner. Row 2 has a NaN candidate (poisons its row).
fn table() -> Candidates {
    Candidates {
        rows: 4,
        count: 3,
        ids: vec![5, 3, 1, 9, 8, 7, 4, 2, 2, 6, 6, 0],
        vals: vec![0.9, 0.8, 0.1, 0.7, 0.6, 0.5, f32::NAN, 0.4, 0.3, 0.25, 0.2, 0.1],
    }
}

/// The engine's greedy sample of row `r` from the FULL table (the reference the gather must match).
fn ref_greedy(c: &Candidates, r: usize) -> u32 {
    let ids: Vec<u32> = c.ids[r * c.count..r * c.count + c.count].iter().map(|&x| x as u32).collect();
    gb_sampling::try_choose(
        &c.vals[r * c.count..r * c.count + c.count],
        &ids,
        r as u64,
        &gb_sampling::Sampling { temperature: 0.0, ..Default::default() },
    )
    .unwrap()
}

#[test]
fn device_verify_gather_rounds_return_the_argmax_column_and_keep_accept_semantics() {
    let mut fake = Fake::default();
    fake.dv = true;
    fake.next_cands = Some(table());
    let mut b = FlashNextBackend::with_runtime(fake, 64, 4, true);
    // anchor + 3 drafts; varied draft lengths across calls exercise the adaptive k path
    let reqs = [RowRequest { slot: 0, first_position: 10, tokens: vec![1, 2, 3, 4] }];
    let outs = b.step_verified(&reqs, 3, &[true]);
    let rows = &outs[0];
    assert_eq!(rows.len(), 4, "one row per input token");
    for (r, row) in rows.iter().enumerate() {
        assert_eq!(row.ids.len(), 1, "the gather returns one candidate per row");
        assert_eq!(row.values.len(), 1);
        if r == 2 {
            assert!(row.values[0].is_nan(), "a NaN candidate must poison its row");
        } else {
            assert_eq!(row.ids[0], ref_greedy(&table(), r), "gathered id must be the greedy winner");
        }
    }
    // accept over the gathered rows, as the engine does: leading drafts equal to the greedy samples
    let drafts = [2u32, 3, 4];
    let accepted = drafts.iter().zip(rows.iter().skip(1)).take_while(|(d, r)| **d == r.ids[0]).count();
    assert_eq!(accepted, 0, "none of these drafts match their argmax");
}

#[test]
fn device_verify_accepted_counts_match_the_host_reference_across_draft_lengths_and_mismatches() {
    // (draft tokens, column-0 winners per draft row, expected accepted count): the winners are what the
    // greedy window would return; a row whose winner differs from its draft is a mismatch
    for (drafts, winners, expect) in [
        (vec![3, 3, 3], vec![3, 3, 3], 3),
        (vec![3, 3, 3], vec![3, 3, 9], 2),
        (vec![3, 3, 3], vec![3, 7, 9], 1),
        (vec![3, 3, 3], vec![7, 7, 9], 0),
    ] {
        let mut c = table();
        for (j, &w) in winners.iter().enumerate() {
            let r = j + 1;
            c.ids[r * c.count] = w as i32;
            c.vals[r * c.count] = 1.0 - j as f32 * 0.1; // keep it the winner of its row
        }
        let mut fake = Fake::default();
        fake.dv = true;
        fake.next_cands = Some(c.clone());
        let mut b = FlashNextBackend::with_runtime(fake, 64, 4, true);
        let tokens: Vec<u32> = std::iter::once(1).chain(drafts.iter().copied()).collect();
        let reqs = [RowRequest { slot: 0, first_position: 10, tokens }];
        let rows = &b.step_verified(&reqs, 3, &[true])[0];
        // host reference: the engine's own greedy samples over the full table, then `accept`
        let ref_samples: Vec<u32> = (1..4).map(|r| ref_greedy(&c, r)).collect();
        let ref_accepted = drafts.iter().zip(&ref_samples).take_while(|(d, s)| **d == **s).count();
        let dv_samples: Vec<u32> = rows.iter().skip(1).map(|r| r.ids[0]).collect();
        assert_eq!(dv_samples, ref_samples, "gathered rows must sample like the full table");
        let dv_accepted = drafts.iter().zip(&dv_samples).take_while(|(d, s)| **d == **s).count();
        assert_eq!(dv_accepted, ref_accepted, "accepted count must match the host reference");
        assert_eq!(dv_accepted, expect);
    }
}

/// The device gather must engage automatically for every all-greedy window, no switch and no
/// environment variable involved — including a stale `REXL3_FLASH_NEXT_DEVICE_VERIFY` value left in
/// the environment, which must not be able to turn it off (the var is no longer read).
#[test]
fn device_verify_engages_automatically_for_all_greedy_batches_and_ignores_the_legacy_env_var() {
    for legacy in ["0", "1"] {
        std::env::set_var("REXL3_FLASH_NEXT_DEVICE_VERIFY", legacy);
        let mut fake = Fake::default();
        fake.dv = true;
        fake.next_cands = Some(table());
        let mut b = FlashNextBackend::with_runtime(fake, 64, 4, true);
        let reqs = [RowRequest { slot: 0, first_position: 10, tokens: vec![1, 2, 3, 4] }];
        let outs = b.step_verified(&reqs, 3, &[true]);
        let rows = &outs[0];
        assert_eq!(rows.len(), 4, "one row per input token");
        for (r, row) in rows.iter().enumerate() {
            assert_eq!(row.ids.len(), 1, "legacy env={legacy}: gather must still engage");
            assert_eq!(row.values.len(), 1);
            if r == 2 {
                assert!(row.values[0].is_nan(), "a NaN candidate must poison its row");
            } else {
                assert_eq!(row.ids[0], ref_greedy(&table(), r), "legacy env={legacy}: gathered id must be the greedy winner");
            }
        }
    }
    std::env::remove_var("REXL3_FLASH_NEXT_DEVICE_VERIFY");
}

#[test]
fn device_verify_falls_back_to_the_full_readback_when_the_runtime_does_not_gather_or_the_batch_is_mixed() {
    // runtime without the gather (Fake dv = false -> None): full candidates come back
    let mut fake = Fake::default();
    fake.next_cands = Some(table());
    let mut b = FlashNextBackend::with_runtime(fake, 64, 4, true);
    let reqs = [RowRequest { slot: 0, first_position: 10, tokens: vec![1, 2, 3, 4] }];
    let outs = b.step_verified(&reqs, 3, &[true]);
    let rows = &outs[0];
    assert_eq!(rows.len(), 4);
    for (r, row) in rows.iter().enumerate() {
        assert_eq!(row.ids.len(), 3, "the fallback must return the full candidate table");
        assert_eq!(row.ids[0], table().ids[r * 3] as u32, "column 0 is the same winner");
    }
    // a mixed batch (one sampled request) also uses the full path, whatever its flags say
    let mut b = FlashNextBackend::with_runtime(Fake::default(), 64, 4, true);
    let reqs = [
        RowRequest { slot: 0, first_position: 10, tokens: vec![1, 2, 3] },
        RowRequest { slot: 1, first_position: 20, tokens: vec![5] },
    ];
    let outs = b.step_verified(&reqs, 2, &[true, false]);
    assert_eq!(outs.len(), 2);
    assert!(outs.iter().all(|rows| rows.iter().all(|row| row.ids.len() == 2)), "mixed batches must keep the full table");
}

#[test]
fn device_verify_errors_are_recorded_and_answer_nan_rows() {
    let mut fake = Fake::default();
    fake.dv = true;
    fake.fail = Some(ModelError::Request("gather failed".into()));
    let mut b = FlashNextBackend::with_runtime(fake, 64, 4, true);
    let reqs = [RowRequest { slot: 0, first_position: 10, tokens: vec![1, 2, 3, 4] }];
    let outs = b.step_verified(&reqs, 3, &[true]);
    assert!(outs[0].iter().all(|row| row.values[0].is_nan()));
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("gather failed"));
}
