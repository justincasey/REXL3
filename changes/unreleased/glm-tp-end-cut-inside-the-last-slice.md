## New features and improvements

- GLM tensor-parallel prompt cache: a long prompt's end cut no longer costs a forward of its own. Under P2 the end cut sits on the 16-row KDA grid, up to 15 rows before the prompt's end. The last slice used to end at the cut, take the snapshot between forwards, and run the remaining rows as one more short forward (126-140 ms on pair A). Now:
  - Rank 0 arms the cut on every rank before the last slice (`ArmCut`, call kind 52), and the slice runs to the prompt's end.
  - Every KDA layer of that forward splits the sequence's chunked recurrence at the cut and copies its conv history out between the two segments. A 16-row boundary moves no bit, and a KDA head-split rank does this on its own heads. There is no sync inside the forward: the second segment is a single piece, so the state at the cut stays in the buffer it did not write.
  - The `Snapshot` call after the forward completes the snapshot: that state buffer, the ring tails, the drafter window, the pages and the seal.
  - Whole slices, forwards of several prompts, forwards in parts and opening forwards all take the cut this way.
  - A restored prompt gives the same bytes as an uncached one. The follow-up turn recomputes the tail rows with its new tokens, as before.
- `REXL3_GLM_TP_INNER_CUT=0` restores the old slicing (the slice ends at the cut). The bits are the same either way.
- Measured on pair A (RED-SNOW 4.91bpw, contract B), one build with the switch off and on:
  - Full harness: prefill 8K 1,549 -> 1,590 / 1,583 tok/s and 32K 1,670 -> 1,686 / 1,675 tok/s. Both runs had gates_failed=0, ref_checked=9 and cache_hit_tokens=8768.
  - `warm,prefill` bench: 8K 1,551 -> 1,594 and 32K 1,671 -> 1,702.
  - Rank 0's last forward of the 8K prompt: 560 + 14 rows (650 + 140 ms) -> 574 rows (661 ms).
