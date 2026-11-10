## New features and improvements

- Qwen3.8-Flash-Next's decode kernels (EXL3 linears, small linear, GDN, gated norm, attention prep, KV write, decode attention, MoE route/gate-up/down, top-k, gate multiply) carry the programmatic-dependent-launch attribute by default, so each launches while its predecessor finishes. Each kernel waits before reading what an earlier kernel wrote, so results are unchanged. Drafted decode measured 62.09 tok/s and serial 33.99 against 60.6-60.9 before (one GB10). `REXL3_GLM_PDL` selects the families.
