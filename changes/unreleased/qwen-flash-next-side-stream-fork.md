## New features and improvements

- Qwen3.8-Flash-Next's decode window runs its independent projections (GDN z next to qkv, attention k/v and the indexer chain next to q, the MTP head's two input projections, the PLE key and value projections) on a second stream, captured as parallel graph branches with their own workspace. Every kernel and its arithmetic are unchanged. Drafted decode measured 61.25 tok/s and serial decode 33.34 tok/s against 58.0-58.5 and 32.2-33.0 before (one GB10). `REXL3_FORK=0` turns it off.
