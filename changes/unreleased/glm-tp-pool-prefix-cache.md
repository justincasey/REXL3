## New features and improvements

- GLM TP=2 serves a prefix cache with `--prefix-cache on`. A follow-up turn that repeats the previous prompt resumes from the whole of it, and its reply is byte-identical to the uncached reply, drafted or serial:
  - The snapshot slots (`REXL3_GLM_PREFIX_ENTRIES`, default one per slot) are planned out of each rank's memory before the KV pool. Rank 0's slots also keep the DFlash2 drafter's last 2,048 context positions, so drafts after a hit match a cold prefill's.
  - Cached prompts keep idle pages of the KV pool, with no pages reserved for the cache. An entry gives its pages back, least recently used first, as soon as an admitted sequence may need them (`rexl3_prefix_cache_evicted_for_pool_total`).
  - With `REXL3_PREFIX_TAIL=0` the snapshot is taken after the prompt's last slice. Under the P2 contract the snapshot sits on the 16-row grid, so a prompt whose length is not a multiple of 16 pays one extra forward of at most 15 rows.
  - Prompts that use the cache, whether restored from a hit or taking a snapshot, share batched prompt forwards and run in parts while other sequences decode, like any other prompt.
- Measured on two GB10s with the RED-SNOW 4.91bpw pack (P2, 4,096-row chunks, fp8 KV, the drafter, 8 entries):
  - Follow-up turn on an 8.9K-token prefix: 8,768 tokens come from the cache, and the first token arrives in 0.48 s instead of 6.84 s.
  - Memory: the entries cost 0.86 GiB on rank 0 (70 MiB of state per entry on each rank, plus the drafter's window on rank 0). With `KV_TOKENS=1179648`, rank 0 keeps at least 4.6 GiB available under the full benchmark.
  - First-turn cost on a ~260-token chat: about 15 ms when the prompt ends on the grid, and 75 to 150 ms when it does not (the extra forward of 2 to 15 rows). The benchmark's short chat goes from 545 to 644 ms to its first token, and prefill 8K from 1,288 to 1,262 tok/s. `REXL3_PREFIX_MIN_CUT` set above a chat's length skips that cost.
  - Prefill 32K and decode speed are unchanged, and replies match the reference.
