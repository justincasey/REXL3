## New features and improvements

- Qwen3.8-Flash-Next's decode MoE route kernel uses one parallel exponential and a shuffle-ordered sum per token, warp reductions for the top-k rounds and a warp-wide grouping of the entries. Routing and every output bit are unchanged (the layer-shape digest test passes against values recorded before the change). The route kernel takes 6.6 us instead of 13.8 us in a microbenchmark and 6.2 us instead of 11.3 us inside the decode graph.
