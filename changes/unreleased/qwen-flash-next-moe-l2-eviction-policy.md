## New features and improvements

- Qwen3.8-Flash-Next's decode MoE kernels stream the expert weights with an evict-first policy and keep the fp32 chunk partials evict-last in L2, so a layer's weight stream no longer pushes the partials out to DRAM and back. Cache policy only; outputs are bit-identical. A microbenchmark layer is 7% faster at 3 rows and 10% faster at 7 rows.
