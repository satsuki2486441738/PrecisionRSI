# SFT collection

Apply the [training state and stages](../methods/training-state.md#stages). Collect every valid target's label, prediction/target position and full-vocabulary-normalized logprob using the native loss mask, including active EOS. Plain CE uses the global valid-token mean of negative logprobs; retain implemented reductions and additional terms.

## LoRA

For adapter training, align initial adapter values, target modules, rank, alpha/scaling, dropout, bias/modules-to-save, dtype, quantization and merged state. Capture every adapter/additional trainable coordinate plus frozen-base identity and unchanged-state checks.

With zero-B initialization, retain present-zero first-step A gradients/updates and changing B coordinates. Use the usual zero-norm metric rules. Capture merged-weight deltas separately when requested.
