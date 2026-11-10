## New features and improvements

- Qwen3.8-Flash-Next's decode hyper-connection pre-mix finishes its k2 stage with a warp-level reduce-scatter instead of redundant exp/divide chains in every lane. Every fp32 operation and its order are kept, so outputs are bit-identical (FNV hash over y, inject and the post-updated streams unchanged for rows 1-8). At 7 rows one pre-mix call takes 67.5 us instead of 91.3 us (20-call graph chain, weights from DRAM, one GB10); at one row 60.6 us instead of 65.0 us.
