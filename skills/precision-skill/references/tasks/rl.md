# RL collection

Apply the [training state and stages](../methods/training-state.md#stages) to actor and critic as used. Replay fixed prompt/response IDs, action positions, masks and EOS/truncation. Capture effective old/reference scores, rewards, advantages, returns and values, marking injected versus recomputed dependencies. Preserve native score transforms, detach behavior and dynamic masks.

Select the algorithm's inputs:

| Algorithm or term | Retain |
|---|---|
| Clipped policy objective | Ratios, bounds and branch decisions; token PPO uses $r=\exp(\log\pi_{current}-\log\pi_{old})$ |
| PPO critic | Current/old values, returns, bootstrap masks and value clipping |
| GRPO | Complete groups, reward components, mean/std, normalization and epsilon |
| KL/entropy | Estimator, direction, coefficient and sampling distribution |

Compare action-logprob MAE/P99/max, native objective differences and requested gradient/update stages. Use dependency or clipping diagnostics to inspect discrepancies.

## Frameworks

Identify the optimizer step, accumulation window, actor/critic order and native entrypoint. Read the selected framework section.

### TRL

Trace batch preparation, action scoring, `compute_loss` and optimizer integration. Capture completion/tool/truncation masks, group normalization and rollout-correction weights. Retain native scoring temperature, dynamic masks, detached old-score handling and ordinary/fused loss selection.

### verl

Preserve sample/group/action mappings through repeats, filtering, balancing, packing, padding removal and rank redistribution. Capture the effective old-policy anchor, correction/rejection weights, reward/KL placement, estimator and loss aggregation.

Use the [training capture methods](../methods/training-state.md#distributed-capture) for distributed tensors and the selected [serving engine](inference.md#serving-engines) for rollout scores.
