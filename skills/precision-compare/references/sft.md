# SFT

## Collection

Select the requested objective, gradient or update stages and their expected parameter/token coordinates. For target-score comparison, collect labels, prediction/target positions and full-vocabulary-normalized logprobs under the native loss mask, including active EOS. Plain CE uses the global valid-token mean of negative logprobs; native loss retains its implemented reduction and additional terms.

### Full-parameter finetuning

Declare the logical parameter inventory for the requested scope; full-model requests include all trainable parameters. Retain tied-weight mappings and trainable flags. At each selected stage, include absent/present-zero gradients and unchanged updates in that inventory.

### LoRA

For adapter training, check initial adapter values, target modules, rank, alpha/scaling, dropout, bias/modules-to-save, dtype, quantization and merged state against the task's common conditions and per-side configuration. Use the requested adapter/additional trainable inventory; retain frozen-base identity and unchanged-state checks.

With zero-B initialization, retain present-zero first-step A gradients/updates and changing B coordinates. Use the usual zero-norm metric rules. Capture merged-weight deltas separately when requested.

### Training stages

Collect the requested stages:

| Stage | Observations |
|---|---|
| Objective | Scores, masks, native losses, coefficients and effective denominators |
| Backward | All present gradients after accumulation, unscale and synchronization; presence inventory |
| Clipping | Native norm/coefficient and postclip gradients when enabled |
| Update | Stored before/after parameters, including unchanged coordinates; applied/skipped event |

Record native parameter/gradient/master storage separately from exported-array and offline-reduction dtypes; preserve logical coordinates. Label independent norm diagnostics as full-model or sampled.

For distributed training, resolve parameter/gradient ownership, synchronization and optimizer storage in the selected backend. Capture accumulated, unscaled, synchronized gradients from native storage (for example, Megatron `main_grad`), postclip gradients before optimizer mutation, and stored parameters after mutation and model copy-back. Record master/state dtypes separately. Map captured shards and replicas using the [coordinate rules](comparison.md#coordinates); use the logical inventory exporter and explicit-view comparator from [comparison](comparison.md).

For optimizer-state or checkpoint-continuation comparisons, include the parameters, moments, step counter and scheduler/RNG/data position needed to establish the requested starting state. Capture cold or restored trajectories as requested; reuse unchanged baselines.

## Framework integration

Use the relevant entrypoint below. Inspect the installed implementation when capture or reduction semantics are unresolved.

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

Resolve the distributed backend as part of framework integration: FSDP/ZeRO shard and optimizer ownership, offload and synchronization; Megatron-style tensor/pipeline/context/data/expert parallel layouts and pipeline stages. Preserve the native training path and map its observations to the selected full-parameter or LoRA inventory.

### Other/custom

Follow [native adaptation](../SKILL.md#adapt-an-unfamiliar-implementation). Locate the framework's effective labels, loss reduction and trainable-parameter storage, then map them to the selected full-parameter or LoRA inventory and training stages.

## Metrics

| Observations | Report |
|---|---|
| Target scores | Each side's NLL and absolute difference; target-logprob MAE, P99 and maximum |
| Native objective | Each side's loss components and absolute differences |
| Requested gradients, postclip gradients and updates | L2 error, RMSE, cosine and maximum absolute error |
| Coverage | Valid targets, paired parameters/elements, gradient presence and optimizer events; for LoRA, adapter coverage and frozen-base checks |

Use [comparison](comparison.md) for score/tensor definitions and tool usage. Native loss keeps its implemented reduction and additional terms.
