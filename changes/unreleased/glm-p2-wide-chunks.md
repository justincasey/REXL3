## New features and improvements

- GLM prefill under the P2 contract takes prompt chunks of up to 8,192 rows (`--prefill-chunk`, any multiple of 16 from 144 to 8,192; the server's default stays 4,096). A wider chunk reads every routed expert once for more rows. P2 rows do not depend on the chunk, so the replies are the same bytes at every width (`reference.json` holds; the new GPU tests check an 8,192-row call against smaller calls for the routed experts and the P2 dense GEMM, and the serving ranks' answers, windows and drafts in chunks of 8,192, 6,144, 4,096 and 512 rows).
  - The start's option check takes the family's own chunk limit (`ModelFamily::max_prefill_chunk`); other families keep 4,096.
- Less scratch for P2 prompt chunks. The KDA qkv rows are now held in the form the P2 GEMM writes them: bf16, and only its own heads on a KDA-split rank. Before, they were P1's fp32 rows of every head. The DFlash2 drafter's context scratch is now sized for one absorb (at most 2,048 rows) instead of a whole chunk. By arithmetic this saves about 0.4 GiB a serving rank at 4,096-row chunks. At 8,192 rows the measured saving was 0.82 / 0.91 GiB on ranks 0 / 1. A model switched to P1 after its build grows the qkv buffer at the switch.
- Measured on two GB10s (pair B), RED-SNOW 4.91bpw, fp8 KV, `KV_TOKENS=1048576` for the wider chunks:

  | Chunk | Prefill 8K (tok/s) | Prefill 32K (tok/s) | stall_ms | Least MemAvailable, rank 0 / rank 1 (GiB) |
  |---|---|---|---|---|
  | 4,096 (KV 1,179,648) | 1,547 | 1,681 | 345 | 5.28 / 5.05 |
  | 7,168 | 1,646 | 1,772 | 339 | 4.33 / 4.31 |
  | 8,192 | 1,644 | 1,777 | 354 | 3.74 / 3.55 |

  At 8,192 rows both ranks drop below a 4 GiB floor.
