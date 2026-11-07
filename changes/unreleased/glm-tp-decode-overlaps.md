## New features and improvements

- GLM-5.3-Flash at TP=2 overlaps more of its decode work, without changing any output bits:
  - A draft round's flush now leaves the last window's KDA state commits (the replay of each layer's accepted rows) on a side stream. They run beside the DFlash2 draft round, which reads none of their buffers. The next forward waits for them before it runs. `REXL3_GLM_COMMIT_SIDE=0` runs them on the main stream as before.
  - In a graphed verify window, each rank's KDA low-rank projections and gates run on a side stream beside the layer's qkv projection (a new layer branch, `REXL3_GLM_OVERLAP` mask 4).
  - Measured on RED-SNOW 4.91bpw on pair B: the one-stream draft call fell from 4.39 to 3.83 ms; a KDA layer from its input norm to the end of its recurrence fell from 220.8 to 216.4 µs at 1 row and from 231.6 to 223.6 µs at 3 rows.
- The cost log's DFlash2 draft lines (`REXL3_GLM_COST_LOG`) now carry each draft's selector margin, so the draft cut can be studied from a log.
