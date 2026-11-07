## New features and improvements

- GLM prefill under the P2 contract is faster, with the same bytes (every kernel's arithmetic unchanged; the KDA tests hold the new path to the old kernels run one by one, the P2 model tests pass):
  - The KDA layers' qkv GEMM writes bf16 rows (prep's own rounding), so prep no longer copies them into the record and the conv shift reads them directly.
  - A KDA layer's prompt pieces are pipelined: piece i+1's projections, prep and conv shift run on a side stream beside piece i's chunked recurrence (a second buffer set, priced as `kda_pipe`; `REXL3_GLM_KDA_PIPE=0` restores the serial order).
  - proj_in and proj_out take shared-memory tiled schedules for 64 rows or more (each output the same chain and tree), and the output gate's second stage runs inside the norm's pass, so z never goes to memory.
  - An mHC site's apply-partials and finalize run in slabs of 256 rows, so the finalize re-reads the streams from L2.
- Measured on pair A (RED-SNOW 4.91bpw, P2, 4,096-row chunks, fp8 KV), rank 0, one 4,096-row chunk: 2,606 -> 2,396 ms (KDA kernels 354 -> 210 ms, mHC 188 -> 158 ms). Quick gate: prefill 8K 1,453 -> 1,535 tok/s, 32K 1,573 -> 1,648 tok/s.
