## New features and improvements

- The Qwen3.8-Flash-Next backend now reports its CUDA-graph caches. `/metrics` carries the `rexl3_graph_cache_*` series for the decode and verify windows and the `rexl3_draft_graph_cache_*` series for the MTP draft chain (replays, captures, eager runs, capture time), and the log gets a `flash_next_graphs` event after every 32 captures with the number of live window and chain graphs, the capture time, the mean drop of free device memory per capture and the resident set of the process.
