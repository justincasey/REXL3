## New features and improvements

- Qwen3.8-Flash-Next's MTP draft chain serves up to four concurrent streams in one pass (per-stream stack segments, one catch-up graph, one head pass and one pick per stream); with two or more streams only the copy drafter ran before. Replies are unchanged (every draft is verified; concurrent == solo holds). Aggregate decode measured 80.1 tok/s on two streams (58.3 before) and 92.6 tok/s on four (78.8 before); one stream is unchanged at 65.7 tok/s (one GB10).
