## New features and improvements

- Under programmatic launch the Qwen3.8-Flash-Next hyper-connection decode kernel `k2t` prefetches its warp's other three weight columns into L2 right after the wait, so their DRAM latency overlaps the low-rank computation. Timing only; outputs are bit-identical. At 7 rows one pre-mix call takes 62.7 us (kHcx on) against 67.7 us off and 91.0 us for the original kernels; at one row 58.1 us against 65.3 us (20-call graph chain, one GB10).
