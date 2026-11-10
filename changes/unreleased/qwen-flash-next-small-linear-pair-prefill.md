## Bug fixes

- Qwen3.8-Flash-Next prefill chunks no longer run the router, shared-gate and GDN `a`/`b` projections through the per-row pair kernel: it cost 6% of prefill throughput once it replaced the many-row kernel there. Chunks of 64 rows or more use the many-row kernel again; decode windows keep the pair kernel. Outputs are bit-identical.
