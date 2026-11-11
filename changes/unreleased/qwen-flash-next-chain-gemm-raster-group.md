## New features and improvements

- The dense chain GEMM of the prefill chunk (attention and delta-net projections, 8,192 rows) walks its output tiles in raster groups of 8 row tiles instead of 16, so the rotated input rows a group re-reads for every column block stay in the L2 beside the streaming weights. The K = 6,144 output projections go from 4.06-4.14 ms to 3.89-3.97 ms (62-63 to 65-66 TFLOP/s), the K = 2,560 projections gain 1-3 %; the order of the output tiles never changes a bit, and the layer outputs are the same bits.
