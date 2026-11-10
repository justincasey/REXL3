## New features and improvements

- The Qwen3.8-Flash-Next hyper-connection decode kernels (`k1u`, `k2t`, `hc_post`) can launch with the programmatic-dependent-launch attribute under a new family bit (`kHcx`, 16 in `REXL3_GLM_PDL`; off by default). They start their first weight loads before the wait, so a call's DRAM stream overlaps the previous kernel's tail. Results are bit-identical with the bit on or off. At 7 rows one pre-mix call takes 64.3 us instead of 67.4 us (20-call graph chain, one GB10).
