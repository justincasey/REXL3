## New features and improvements

- Qwen3.8-Flash-Next packs that store their hyper-connection, router, shared-expert gate, GDN `a`/`b` and PLE projection tensors as bf16 now load: the loader rounds them to fp16, to nearest even, which is what the kernels read. On the pack this was measured on, 567 tensors (753,297,920 elements) went through it and 104,786 elements (0.014%) changed, all with an absolute value below 6.1e-5; none overflow. Packs that store fp16 are unaffected.

## Known issues

- The bf16 to fp16 rounding at load is a numeric step for the packs it applies to. An exact alternative would read bf16 in the kernels that consume these tensors; it is not implemented.
