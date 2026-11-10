## New features and improvements

- Qwen3.8-Flash-Next's verify window is captured as two graphs split at the PLE stage, so the host's n-gram row gather (0.7-1.2 ms a drafted window) overlaps the GPU running the layers before it. Kernels, order and arguments are unchanged and the replies are the pinned ones. Drafted decode measured 60.85 and 60.60 tok/s against 58.0-58.7 before on one GB10.
