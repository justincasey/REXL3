## New features and improvements

- GLM prefill under the P2 contract is faster again, with the same bytes:
  - KDA proj_in (64 rows or more) runs proj_in_kernel's own lane structure, 8 rows x 8 outputs a lane from shared-memory tiles staged once (`proj_in_lane_kernel`); it replaces the 64 x 32 tiled schedule, which was as fast alone but competed with the chunked recurrence for shared-memory bandwidth on the pipelined side stream.
  - A row-split TP rank's routed-expert part: the rows the peer reads are summed first and swapped while the shared expert runs, the rank's own rows after it, so the part swap no longer waits behind the whole combine.
- Measured on pair A (RED-SNOW 4.91bpw, P2, fp8 KV), rank 0 per-chunk times of salted 32K prompts, two boots x 18 chunks per build, interleaved: 2,441 -> 2,425 ms (swap order) -> 2,400 ms (lane proj_in) a 4,096-row chunk.
