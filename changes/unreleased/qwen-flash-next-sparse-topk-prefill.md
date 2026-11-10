## New features and improvements

- Qwen3.8-Flash-Next prompt prefill selects each query's 512 attended blocks faster, with the same blocks in the same order (ties at the threshold still go to the lower index): the radix select builds one block-wide histogram a pass instead of 32 per-warp ones, scans its 256 bins with one warp instead of one thread, and combines the warps' counts of the final ordered pass with shuffles, one barrier a pass instead of four. On the last chunk of a 32K prompt (4,096 rows, 7,000 to 8,000 blocks each) the selection takes 1.39 ms per layer, against 3.5 ms before.
