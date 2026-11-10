## New features and improvements

- Qwen3.8-Flash-Next's speculation cost model is a measured (streams, rows) grid for one to four streams, including the step past 8 rows (a ninth row costs 7 to 9 ms more than the eighth at two or more streams, then about 3.5 ms a row) and the extra chain cost of each further stream. Replies are unchanged; decode and concurrency screens are within noise of before (65.6 tok/s single stream; 80.8 and 92.5 tok/s on two and four).
