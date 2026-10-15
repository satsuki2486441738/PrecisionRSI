# TRL and verl

Apply the [RL protocol](training.md#rl) to each framework's GPU/NPU pair. Inspect the installed trainer, loss variant and backend; hooks and supported algorithms vary by release.

## Update boundary

Replay the fixed trajectory/dependencies into native loss, backward and optimizer execution. Identify one actual optimizer step per active model, its accumulation window and actor/critic order; a rollout batch or epoch may contain several steps. Scheduled inactive models remain outside that step's coverage.

Use backend-aware gradient/parameter access for FSDP/ZeRO/Megatron; callbacks may not expose synchronized full gradients. Follow [training collection](training.md#common-training-state) for stages and coverage. Record actual integration scope, such as a loss adapter versus the complete trainer loop.

## TRL

Trace batch preparation, action scoring, loss and optimizer integration. Capture effective loss inputs, including completion/tool/truncation masks, group normalization and rollout-correction weights. Preserve dynamic masks as intermediates; numerical branch changes are not missing trajectory coordinates.

Old scores may be a retained snapshot or detached current scores. Preserve that rule and detach semantics rather than substituting a common denominator. Verify temperature and other score transforms used by the trainer; raw inference scores may define a different objective. Observe the selected native or fused loss path.

## verl

Trace rollout, reward/reference/value computation, advantages and actor/critic update workers. Retain sample/group/action mappings through repeats, filtering, balancing, packing, padding removal and rank redistribution.

Distinguish rollout scores from a recomputed old-policy anchor; some configurations deliberately reuse rollout scores. Preserve correction/rejection weights, reward/KL placement, estimator and loss aggregation. Capture actual training-backend observations; RPC averages cannot supply full gradients or step objectives.

For vLLM/SGLang rollout diagnosis, read [engines](engines.md). Fixed-trajectory trainer comparison does not require regenerating rollout or adding inference metrics.
