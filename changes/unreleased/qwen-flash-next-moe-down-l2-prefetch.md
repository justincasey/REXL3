## New features and improvements

- Qwen3.8-Flash-Next's decode MoE down kernel prefetches its warp's expert weights into L2 before its programmatic-launch wait, so they stream while the gate/up stage finishes. Timing only; outputs are bit-identical.
