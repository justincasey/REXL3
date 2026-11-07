## New features and improvements

- GLM TP=2 serves concurrent users better, and every reply keeps the bytes it gets alone, under both prefill contracts:
  - Prompts that arrive together go through one prefill forward. Under P1 each prompt runs its own row-by-row KDA recurrence on its own slot. An idle engine waits 15 ms for requests sent together (`REXL3_PREFILL_GATHER_MS`).
  - While sequences decode, a step runs only one prefill forward. That forward holds either fresh short prompts, up to `REXL3_GLM_PREFILL_BATCH_ROWS` rows (default 2,048), or one part of a long slice. The two take turns.
  - A long slice runs in layer parts of about `REXL3_GLM_PREFILL_PART_MS` (default 100 ms), with a decode round between parts.
  - Window shapes are captured into CUDA graphs on their first sighting (`REXL3_GLM_GRAPH_AT=1` keeps the second).
- Measured on two GB10s with the RED-SNOW 4.91bpw pack, under P1 with the fp16 KV cache, compared with the previous build:
  - C=8: 72.0 -> 88.5 tok/s; worst time to first token 6.30 -> 2.70 s.
  - A short chat's time to first token under load: 2.85 -> 1.03 s.
  - Mean trace latency: 88.9 -> 34.1 s.
  - The longest stall of decoding streams: 1.32 -> 0.81 s.
  - Single-prompt prefill and decode are unchanged, and replies match the unmodified engine's reference.
