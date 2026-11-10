## New features and improvements

- Qwen3.8-Flash-Next prompt prefill is faster with every output bit unchanged (serial replies, prefill and needle replies, drafted == serial, cached == uncached). The hyper-connection pre-mix runs its split-K products as a few large cublasLt calls with the library's own M = 64 algorithm family; the router, shared gate and GDN `a`/`b` projections use a weight-stationary many-row kernel on chunks of 64 rows or more; prefill attention serves four query heads a warp; the dense prefill linears run the fused chain GEMM.
