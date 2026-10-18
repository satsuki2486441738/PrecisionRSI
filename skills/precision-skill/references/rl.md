# RL collection

Apply the [common training stages](training.md#common-training-state) to actor and critic as used. Replay fixed prompt/response IDs, action positions, masks and EOS/truncation. Capture effective old/reference scores, rewards, advantages, returns and values, marking injected versus recomputed dependencies. Preserve native score transforms, detach behavior and dynamic masks.

Select the algorithm's inputs:

| Algorithm or term | Retain |
|---|---|
| Clipped policy objective | Ratios, bounds and branch decisions; token PPO uses $r=\exp(\log\pi_{current}-\log\pi_{old})$ |
| PPO critic | Current/old values, returns, bootstrap masks and value clipping |
| GRPO | Complete groups, reward components, mean/std, normalization and epsilon |
| KL/entropy | Estimator, direction, coefficient and sampling distribution |

Compare action-logprob MAE/P99/max, native objective differences and requested gradient/update stages. Use dependency or clipping diagnostics to inspect discrepancies. Read the selected [TRL/verl integration](rl-frameworks.md) when preparing a native collector.
