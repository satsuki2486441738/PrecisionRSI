# Native RL integration

Use the [fixed-trajectory protocol](rl.md). Identify the optimizer step, accumulation window, actor/critic order and native entrypoint. Read the selected framework section.

## TRL

Trace batch preparation, action scoring, `compute_loss` and optimizer integration. Capture completion/tool/truncation masks, group normalization and rollout-correction weights. Retain native scoring temperature, dynamic masks, detached old-score handling and ordinary/fused loss selection.

## verl

Preserve sample/group/action mappings through repeats, filtering, balancing, packing, padding removal and rank redistribution. Capture the effective old-policy anchor, correction/rejection weights, reward/KL placement, estimator and loss aggregation.

Use [backend hooks](training-backends.md) when implementing tensor capture. Read the selected [serving engine](engines.md) when collecting rollout scores through it.
