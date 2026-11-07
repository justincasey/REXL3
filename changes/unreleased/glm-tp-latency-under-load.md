## New features and improvements

- GLM TP=2 keeps decoding streams moving while prompts fill, and gathers bursts more reliably; every reply keeps the bytes it gets alone:
  - While sequences decode, a short prompt's forward runs in parts of its own of about `REXL3_GLM_PREFILL_SHORT_PART_MS` (default 300 ms), on top of a long slice in parts if one is paused (a model keeps two forwards paused). The decode round between its parts drafts nothing, so its first token waits as little as possible.
  - A long slice's parts follow each layer's own measured time, so no part runs two layers past the target after cheap ones.
  - The prefill pass-over guard counts a slice in parts as one slice: when a slice ends, the prompt with the fewest rows left goes next, instead of two long prompts taking turns slice by slice.
  - With nothing decoding, the first forward of a burst runs its first ~50 ms of layers and pauses (`REXL3_GLM_PREFILL_OPEN_MS`); prompts of the burst that reached the engine meanwhile join it in one forward instead of waiting for it.
- Measured on two GB10s (pair B) with the RED-SNOW 4.91bpw pack under contract B (P2 prefill, fp8 latent KV, prompt cache), against the previous build:
  - Longest gap between a decoding stream's tokens while a 32K prompt fills: 623 -> 365 ms.
  - Mean latency of the bench's open-loop trace: 27.6 -> 27.2 s (its 8K prompt's first token 33 -> 27 s in a probe run).
  - Time to first token of a 32K prompt and of a short chat under load, burst throughput and worst first-token time, single-prompt prefill and decode are unchanged within run-to-run noise; replies match the pinned reference.
