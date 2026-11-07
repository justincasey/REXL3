## New features and improvements

- GLM prefill under the P2 contract is faster, with the same bytes (every pair's arithmetic unchanged; the P2 expert golden hashes and `reference.json` hold):
  - The routed experts' work items (P1's items of 1 to 4 tiles of one expert) are listed in expert order and run in one launch per GEMM. P1's three item classes each had their own launch, so an expert with a remainder item read its weights from memory twice; now its items run side by side and share the weights through L2.
  - A layer's subbanks (mixed expert widths) run each stage as one launch over the layer's tables by global expert id: one plan, one input rotation, one gate/up and one down GEMM whose blocks switch to their expert's width. A split gate/up subbank still runs the decode path's windows.
  - The input rotation is one block per hidden row, read once for all of its pairs (the plan leaves each pair's sorted row).
- Measured on two GB10s (pair A) with the RED-SNOW 4.91bpw pack, P2, 4,096-row chunks, fp8 KV, against b665a93 (quick gate):
  - Prefill 8K: 1,360 -> 1,458 tok/s. Prefill 32K: 1,459 -> 1,571 tok/s. Short-prompt TTFT: 522 -> 470 ms.
