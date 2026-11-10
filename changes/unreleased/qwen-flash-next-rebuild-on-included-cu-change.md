## Bug fixes

- Building the CUDA kernels no longer reuses a stale object when only a `.cu` file that another kernel source includes has changed (for example `gdn_hv32.cu` or `attention_kv2.cu`). The build script now compares each source's object with the `.cu` files it includes, so an incremental build picks up an edit to an included kernel file.
