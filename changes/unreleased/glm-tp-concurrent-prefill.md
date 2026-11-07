## New features and improvements

- GLM TP=2 under the P2 contract serves concurrent users better, and every reply keeps the bits it gets alone:
  - Prompts that arrive together go through one P2 forward. An idle engine waits 15 ms for requests sent together (`REXL3_PREFILL_GATHER_MS`).
  - While sequences decode, a step runs only one prefill forward. That forward holds either fresh short prompts, up to `REXL3_GLM_PREFILL_BATCH_ROWS` rows (default 2,048), or one part of a long slice. The two take turns.
  - A long slice runs in layer parts of about `REXL3_GLM_PREFILL_PART_MS` (default 100 ms), with a decode round between parts.
  - Window shapes are captured into CUDA graphs on their first sighting (`REXL3_GLM_GRAPH_AT=1` keeps the second).
- Measured on two GB10s with the RED-SNOW 4.91bpw pack, compared with the previous build:
  - C=8: 67.3 -> 89.1 tok/s; worst time to first token 5.35 -> 1.51 s.
  - A short chat's time to first token under load: 1.92 -> 0.92 s.
  - Mean trace latency: 65.3 -> 28.3 s.
  - Single-prompt prefill and decode are unchanged.
