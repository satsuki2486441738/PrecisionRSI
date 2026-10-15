# TRL and verl

Apply the [RL protocol](training.md#rl) separately to each framework's GPU/NPU pair. Cross-framework equivalence is another experiment. Reuse the approved algorithm, installed release/backend and loss variant; inspect their actual implementation before choosing hooks. Trainer/worker names and supported algorithms change between releases. This guide defines collection boundaries, not installed adapters or device support.

## Fix the replay boundary

Replay the fixed trajectory and dependency bundle at the update boundary, preserving native loss, backward, communication and optimizer execution. Save the effective loss inputs and their provenance, including masks, coefficients, reduction, policy versions and any importance-sampling correction. Mark locally recomputed dependencies separately; do not silently replace them with common injected values. Retain stop-gradient/detach semantics: identical forward values do not establish identical gradients.

A rollout batch, trainer iteration or PPO epoch can contain multiple optimizer updates. Identify one actual optimizer step per trained model, its complete accumulation window and actor/critic order. Restore the same starting state on both sides; record skipped steps. If a model is not updated under the configured schedule, report it as outside that step's coverage rather than forcing an update. Capture full gradients after accumulation/unscale/communication and before clipping, post-clip when used, and actual stored parameters immediately before/after the step. Use backend-aware access for FSDP/ZeRO/Megatron shards or master weights; do not assume a trainer callback sees complete synchronized gradients.

Retain each microstep's objective components, denominators and effective backward scaling; derive a step objective only from the implementation's actual accumulation/distributed reduction, never an unweighted mean of rank logs. A numerically triggered skipped step is an event to report: preserve comparable finite scores/losses and actual before/after values, label the update as skipped versus applied, and block only missing, nonfinite or semantically incompatible metric scopes; do not present it as agreement between two applied updates.

## TRL

Inspect the selected trainer's batch preparation, action-logprob computation, loss implementation and optimizer integration (including Transformers/Accelerate or other backends where used). Capture the actual tensors consumed by the loss, not only logged batch averages. For GRPO, preserve completion masks plus tool/truncation or other objective masks, group membership, reward scaling, advantage normalization, loss variant and token/sequence weighting. Track dynamically computed masks as intermediates; distinguish numerical branch changes from missing trajectory coverage.

Distinguish generation/behavior scores, old-policy scores and reference scores. Some configurations derive old scores from detached current scores, while others retain a snapshot across multiple updates; preserve the configured rule instead of injecting an arbitrary common denominator. When rollout correction is enabled, retain its weights/masks and clipping semantics. Verify the installed action-score temperature and other transforms; do not replace the trainer objective with raw inference scores. PPO/critic collection applies only if the chosen installed trainer actually implements it.

## verl

Trace the selected trainer's trajectory batch through rollout, reward/reference/value computation, advantage construction and actor/critic update workers. Preserve stable sample/group/action coordinates across repeats, filtering, balancing, packing, padding removal, rank redistribution and microbatches. Recover logical positions and parameters from explicit mappings; worker output order is not an identity.

Distinguish rollout logprobs from the old-policy anchor recomputed by the trainer. Some configurations intentionally reuse rollout scores as the anchor; record the actual choice and any correction/rejection weights or masks. Preserve advantage estimator, reward/KL placement, policy/value loss variants, loss aggregation and effective global denominators. Capture losses and gradients inside the actual training backend's accumulation/optimizer boundary; RPC-level metrics alone do not provide full observations. Keep actor and critic updates and provenance separate.

## Rollout and reporting

If rollout uses vLLM or SGLang, read [engines](engines.md) only when diagnosing that producer. Fixed-trajectory trainer comparison does not require rerunning generation or adding inference top-100 metrics. A rollout/trainer score check needs the same policy identity, trajectory and declared sampling transforms; mismatch may reflect semantics rather than arithmetic.

Report only action-score errors, objective differences, and gradient/actual-update errors by default. Keep algorithm-specific evidence available for targeted localization. CPU fixtures establish validator/arithmetic behavior; installed-framework execution and GPU/NPU agreement require their own evidence.
