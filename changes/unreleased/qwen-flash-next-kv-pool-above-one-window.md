## Bug fixes

- Qwen3.8-Flash-Next no longer caps an explicit `--kv-tokens` at one context window. A profile with four slots of 262,144 tokens can now ask for a pool of 1,081,344 tokens (four windows and 128 padding pages) and gets it; four concurrent prompts of 238,181 to 251,042 tokens (981,989 in all) were served from such a pool and each answered with its own needle. Without `--kv-tokens` the pool is still one window.
