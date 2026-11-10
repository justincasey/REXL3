## New features and improvements

- Qwen3.8-Flash-Next prices speculative rounds from the measured cost of a verify window (28.6 / 32.5 / 36.0 / 39.1 / 42.7 / 46.1 / 49.5 ms for 1 to 7 rows, the draft chain 0.28 + 1.2 ms a step) instead of a hand-coded table, and trims the window by the head's draft probabilities. Replies are unchanged (greedy verify). `/metrics` gains `rexl3_round_rows_total`, `rexl3_round_rows_verify_seconds_total` and `rexl3_round_rows_draft_seconds_total`, labelled by streams and rows. `REXL3_SPEC_BUDGET` and `REXL3_SPEC_LEARN` select the policy.
