## New features and improvements

- Decode got faster without changing any output bits. Both the routed-expert GEMV and the EXL3 linear decode pass (up to 16 rows) now copy each warp's weight tiles and x words into shared memory with `cp.async`, several k steps ahead of the step being computed. Each warp has 6 KB of shared memory for this. Before, a warp prefetched only the next step's weights into registers and loaded x at each step. The kernels use at most 128 registers, which fits 4 blocks of 4 warps per SM. Before the PDL wait, the expert GEMV now starts copying its first weight steps, and the linear kernel its first steps.
- At TP=2, the dense MLPs of layers 0–2 are now split between the ranks in decode windows. They are cut the same way as the shared expert: each rank computes the gate/up strips of its half of the intermediate width and the down-projection K-split partials for that half. The ranks then swap their partials, and both ranks finish all splits in order. This is a new `REXL3_GLM_TP_SPLIT` member, `dense`, which is on by default. Prefill chunks still run the dense MLPs replicated.
- Measured on RED-SNOW 4.91bpw at TP=2 (pair A), with serial token ids unchanged and drafted output equal to serial:
  - 1-row window: 36.0 → 34.5 ms.
  - 3-row window: 54.4 → 51.9 ms.
  - Drafted prose: 33.1 → 34.7–35.1 tok/s.
