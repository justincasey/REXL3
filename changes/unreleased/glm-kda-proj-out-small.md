## New features and improvements

- KDA decode windows (GLM-5.3-Flash, Ling-3.1) run the low-rank gate's second stage (`proj_out`: f_b / g_b and the gates) below 64 rows with a kernel that stages each block's weight rows in shared memory through coalesced loads. The old kernel had each thread read its own 256-byte weight row, so every warp load touched 32 cache lines. Output bits are unchanged: each output is the same 128-term `fmaf` chain in the same order, with the same epilogue.
  - On GB10 with cold weights (64 heads, 1 row), the stage fell from 31.4 to 22.4 µs per layer.
  - Measured on RED-SNOW 4.91bpw at TP=2 on pair A, rank 0 under nsys: a KDA layer from its input norm to the end of its recurrence fell from 217 to 205 µs at 1 row and from 225 to 214 µs at 3 rows. In the cost log (without nsys), the 1-row window is 34.26 ms and the 3-row window 50.75 ms.
