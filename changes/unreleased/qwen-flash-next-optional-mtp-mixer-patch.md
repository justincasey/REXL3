## Bug fixes

- Qwen3.8-Flash-Next packs that keep the three MTP hyper-connection mixer tensors inside their shards no longer need a separate `mtp_hyper_connection_mixer_patch.safetensors`. The loader and the pack fingerprint skip a missing patch file, and the binder still names any mixer tensor that neither the shards nor the patch file provide. Packs that ship the patch file load as before.
