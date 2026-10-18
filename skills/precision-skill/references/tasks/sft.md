# SFT collection

Apply the [training state and stages](../methods/training-state.md#stages). Collect every valid target's label, prediction/target position and full-vocabulary-normalized logprob using the native loss mask, including active EOS. Plain CE uses the global valid-token mean of negative logprobs; retain implemented reductions and additional terms.

## Full-parameter finetuning

Collect the complete logical model-parameter inventory, retaining tied-weight mappings and trainable flags. Capture every trainable parameter's gradient and stored update for the requested stages, including absent/present-zero gradients and unchanged updates.

## LoRA

For adapter training, align initial adapter values, target modules, rank, alpha/scaling, dropout, bias/modules-to-save, dtype, quantization and merged state. Capture every adapter/additional trainable coordinate plus frozen-base identity and unchanged-state checks.

With zero-B initialization, retain present-zero first-step A gradients/updates and changing B coordinates. Use the usual zero-norm metric rules. Capture merged-weight deltas separately when requested.

## Framework integration

### PyTorch training loops

Locate native batch preparation, model/loss evaluation and optimizer boundaries. Preserve the implemented label shift, loss mask, accumulation and reduction. Use [distributed capture](../methods/training-state.md#distributed-capture) for sharded training.

### Other/custom

Follow [native adaptation](../methods/collection.md#adapt-an-unfamiliar-implementation). Locate the framework's effective labels, loss reduction and trainable-parameter storage, then map them to the selected full-parameter or LoRA inventory and training stages.

## Metrics

| Observations | Report |
|---|---|
| Targets and objective | Each side's NLL/native loss components and their absolute differences; target-logprob MAE, P99 and maximum |
| Requested gradients, postclip gradients and updates | L2 error, RMSE, cosine and maximum absolute error |
| Coverage | Valid targets, paired parameters/elements, gradient presence and optimizer events; for LoRA, adapter coverage and frozen-base checks |

Use the [score and tensor definitions](../methods/metrics.md) and select the [tools](../methods/tools.md) for the captured stages. Native loss keeps its implemented reduction and additional terms.
