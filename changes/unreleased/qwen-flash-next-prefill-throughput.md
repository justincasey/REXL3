## New features and improvements

- Qwen3.8-Flash-Next prompt prefill is faster with every output bit unchanged (serial replies, prefill and needle replies, drafted == serial, cached == uncached). The router, shared gate and GDN `a`/`b` projections use a weight-stationary many-row kernel on chunks of 64 rows or more.
