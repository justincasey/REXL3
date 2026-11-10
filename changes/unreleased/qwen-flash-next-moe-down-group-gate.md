## New features and improvements

- Qwen3.8-Flash-Next's decode MoE down warps wait for their own group's activation (a per-group counter the gate/up stage bumps per finished column block) instead of the whole gate/up grid. Outputs are bit-identical. A microbenchmark layer is 3.5% faster at 3 rows and 3.7% at 7 rows.
