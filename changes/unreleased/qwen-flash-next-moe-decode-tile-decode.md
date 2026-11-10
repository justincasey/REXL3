## New features and improvements

- Qwen3.8-Flash-Next's decode MoE kernels build their MMA fragments through half2 arithmetic and batch the loads of the gate/up and down epilogues and of the top-k combine. Every output bit is unchanged; a new layer-shape test pins a digest of the layer's output at 1 to 7 rows against values recorded before the change.
