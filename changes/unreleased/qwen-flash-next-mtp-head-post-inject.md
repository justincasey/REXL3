## Bug fixes

- Qwen3.8-Flash-Next's MTP draft head applies its decoder layer's pending hyper-connection post-inject before it collapses its stream stack. Without it the drafts were built from an incomplete stack and the acceptance of drafted tokens fell from 2.89 to 1.45 tokens a round (replies were never affected: every draft is verified); with it acceptance is back to the head's value.
