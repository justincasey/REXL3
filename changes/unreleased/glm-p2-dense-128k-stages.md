## New features and improvements

- GLM prefill under the P2 contract: the dense EXL3 GEMM's 128 x 1 tile (the pick for the big layers: KDA qkv and o, MLA o, the dense MLP) streams 128 k a stage through a 2-stage cp.async ring, one barrier a 128-k fold block, instead of 64 k through 3 stages. Same MMAs in the same order: the same bytes. MEASURED on GB10 (`bench_p2_on_the_serving_shapes`, cold, CUDA events): KDA qkv 3.30 -> 3.13 ms, KDA o 2.40 -> 2.27, MLA o 4.7-4.9 -> 4.4-4.5, dense gate/up 3.28 -> 3.12 a launch.
