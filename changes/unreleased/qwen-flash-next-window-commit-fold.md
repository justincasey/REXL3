## New features and improvements

- Qwen3.8-Flash-Next's decode window replays the rows of the recurrent state its commit owes inside the next window kernel instead of in standalone commit kernels, and no longer syncs the host before the draft chain. Replies are unchanged; serial decode measured 32.96 and 32.75 tok/s against 32.49 and 32.46 on two GB10s, and drafted decode is flat (58.0 tok/s either way). The memory plan accounts for the second record set (+47.6 MiB at 64 rows).
