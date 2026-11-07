## New features and improvements

- GLM TP=2 prefill under the P1 contract is faster, with the same bytes (the routed-expert width split, chunks of 256 rows or more):
  - Each rank finishes only its own rows of a MoE layer. It swaps half of the pair buffer with its peer (the peer's rows of its K-split partials go, its own rows of the peer's come), adds the two partials in the down epilogue (the all-reduce's exact two-operand add), combines, and swaps the finished output rows. Half the pair buffer crosses each way instead of the all-reduce of the whole buffer, and the finish and the shared expert run on half the rows.
  - The prefill MoE plan of every subbank (and of an owned bank) is built by the multi-block planner, which writes the single-block planner's bytes.
- Measured on two GB10s (pair B) with the RED-SNOW 4.91bpw pack, P1, fp16 KV, 2,048-row chunks, against the previous build:
  - Prefill 8K: 657 -> 773 tok/s. Prefill 32K: 664 -> 791 tok/s.
  - Decode, KV tokens and boot are unchanged, and replies match the unmodified engine's reference.
