# RL collection

Apply the [training state and stages](../methods/training-state.md#stages) to actor and critic as used. Replay fixed prompt/response IDs, action positions, masks and EOS/truncation. Capture effective old/reference scores, rewards, advantages, returns and values, marking injected versus recomputed dependencies. Preserve native score transforms, detach behavior and dynamic masks.

Select the algorithm's inputs:

| Algorithm or term | Retain |
|---|---|
| Clipped policy objective | Ratios, bounds and branch decisions; token PPO uses $r=\exp(\log\pi_{current}-\log\pi_{old})$ |
| PPO critic | Current/old values, returns, bootstrap masks and value clipping |
| GRPO | Complete groups, reward components, mean/std, normalization and epsilon |
| KL/entropy | Estimator, direction, coefficient and sampling distribution |

Use dependency or clipping diagnostics to inspect discrepancies.

## Framework integration

Identify the optimizer step, accumulation window, actor/critic order and native entrypoint. Read the selected framework section.

### TRL

Trace batch preparation, action scoring, `compute_loss` and optimizer integration. Capture completion/tool/truncation masks, group normalization and rollout-correction weights. Retain native scoring temperature, dynamic masks, detached old-score handling and ordinary/fused loss selection.

### verl

Preserve sample/group/action mappings through repeats, filtering, balancing, packing, padding removal and rank redistribution. Capture the effective old-policy anchor, correction/rejection weights, reward/KL placement, estimator and loss aggregation.

Use the [training capture methods](../methods/training-state.md#distributed-capture) for distributed tensors and the selected [serving engine](inference.md#framework-integration) for rollout scores.

### Other/custom

Follow [native adaptation](../methods/collection.md#adapt-an-unfamiliar-implementation). Locate native trajectory preparation, action scoring, objective evaluation and actor/critic updates. Preserve sample/group mappings and effective masks, then capture the algorithm inputs and training stages above.

## Metrics

| Observations | Report |
|---|---|
| Action scores | Logprob MAE, P99 and maximum at paired valid actions |
| Native objective | Each side's loss components and absolute differences; retain reward/advantage/value inputs for localization |
| Requested gradients, postclip gradients and updates | L2 error, RMSE, cosine and maximum absolute error |
| Coverage | Paired trajectories/groups/actions, effective masks, parameter/element counts and optimizer events |

Use the [score and tensor definitions](../methods/metrics.md). Select the task adapter for native scores/objectives and the [tensor tools](../methods/tools.md) for training stages.
