## New features and improvements

- Qwen3.8-Flash-Next prompt prefill no longer writes and re-reads the sparse attention's partial sums: a prefill chunk's attention kernel (one part) ends with the merge's own arithmetic and stores the bf16 output directly, so the merge kernel and the 100 MB of fp32 partial sums per layer are gone. The output is the same bits as the merge kernel's (hash-identical on a 4,096-row chunk and equal to the one-row kernel's row by row). Attention plus merge of a 4,096-row chunk takes 24.7 ms per layer, against 25.1 ms before.
