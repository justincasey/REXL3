## New features and improvements

- Qwen3.8-Flash-Next's decode hyper-connection pre-mix loads the window's x rows once per chunk for nine weight rows instead of once per row, and runs the 32-wide reductions as one reduce-scatter. Outputs are bit-identical (same FNV hash as the reference kernels). At 7 rows the L2-warm call takes 43.3 us instead of 53.2 us and the 20-call graph chain 66.1 us instead of 67.5 us; one row is within the run-to-run spread (one GB10).
