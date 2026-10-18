# SFT

## Collection

Apply the [training state and stages](../methods/training-state.md#stages). Collect every valid target's label, prediction/target position and full-vocabulary-normalized logprob using the native loss mask, including active EOS. Plain CE uses the global valid-token mean of negative logprobs; retain implemented reductions and additional terms.

### Full-parameter finetuning

Collect the complete logical model-parameter inventory, retaining tied-weight mappings and trainable flags. Capture every trainable parameter's gradient and stored update for the requested stages, including absent/present-zero gradients and unchanged updates.

### LoRA

For adapter training, align initial adapter values, target modules, rank, alpha/scaling, dropout, bias/modules-to-save, dtype, quantization and merged state. Capture every adapter/additional trainable coordinate plus frozen-base identity and unchanged-state checks.

With zero-B initialization, retain present-zero first-step A gradients/updates and changing B coordinates. Use the usual zero-norm metric rules. Capture merged-weight deltas separately when requested.

## Framework integration

Select the native entrypoint and inspect the installed version's implementation.

| Framework / entrypoint | Inspect in the installed implementation |
|---|---|
| PyTorch training loop | Batch preparation, model/loss call and optimizer boundaries; retain native label shift, mask, accumulation and reduction |
| [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) | Data collator, `compute_loss`, `training_step` and optimizer integration; resolve effective labels and accumulation denominators |
| [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) | Dataset formatting/packing, completion or assistant masks, loss path and PEFT trainable state |
| [LLaMAFactory](https://github.com/hiyouga/LlamaFactory) | Native SFT workflow, resolved template, full/LoRA configuration and the resulting trainer/model |
| [ms-swift](https://github.com/modelscope/ms-swift) | Native SFT entrypoint, selected training backend, resolved template/loss settings and tuner state |
| [Axolotl](https://docs.axolotl.ai/) | Resolved training configuration, preprocessing/packing, effective labels and adapter state |
| [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/peft.html) | Native finetuning entrypoint, checkpoint/adapter conversion and logical tensor mapping |
| [MindSpeed-LLM](https://github.com/Ascend/MindSpeed-LLM) | Native finetuning workflow, selected backend, checkpoint conversion and parameter/gradient storage |

Resolve the actual distributed backend separately using [training-state capture](../methods/training-state.md#distributed-capture). Preserve each framework's native training path and map its observations to the same full-parameter or LoRA inventory.

### Other/custom

Follow [native adaptation](../methods/collection.md#adapt-an-unfamiliar-implementation). Locate the framework's effective labels, loss reduction and trainable-parameter storage, then map them to the selected full-parameter or LoRA inventory and training stages.

## Metrics

| Observations | Report |
|---|---|
| Targets and objective | Each side's NLL/native loss components and their absolute differences; target-logprob MAE, P99 and maximum |
| Requested gradients, postclip gradients and updates | L2 error, RMSE, cosine and maximum absolute error |
| Coverage | Valid targets, paired parameters/elements, gradient presence and optimizer events; for LoRA, adapter coverage and frozen-base checks |

Use the [score and tensor definitions](../methods/metrics.md) and select the [tools](../methods/tools.md) for the captured stages. Native loss keeps its implemented reduction and additional terms.
