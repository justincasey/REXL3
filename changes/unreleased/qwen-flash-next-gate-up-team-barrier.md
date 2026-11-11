## New features and improvements

- The routed experts' gate/up launch of a prefill chunk (Flash-Next, 512 experts, 8,192 rows) lets its two teams of eight warps, gate and up, run their pipeline stages independently: each team waits at its own named barrier instead of the block-wide one, so one team's weight decode overlaps the other's tensor-core work. The layer's gate/up kernel takes 8.78-8.88 ms instead of 9.41-9.44 ms (routing spread 0.8; 9.13-9.15 ms instead of 9.69-10.23 ms at 1.4), and every output is bit for bit the same. `REXL3_MOE_PF_TEAMBAR=0` restores the block-wide barrier.
