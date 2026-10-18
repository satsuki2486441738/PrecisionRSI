# RL

## Collection

Apply the [training state and stages](sft.md#training-stages) to actor and critic as used. Replay fixed prompt/response IDs, action positions, masks and EOS/truncation. Capture effective old/reference scores, rewards, advantages, returns and values, marking injected versus recomputed dependencies. Preserve native score transforms, detach behavior and dynamic masks.

Identify the optimizer step, accumulation window and actor/critic order.

Capture the inputs, configuration and intermediate values used by the native objective, as applicable:

| Objective component | State to capture |
|---|---|
| Clipped policy objective | Ratios, bounds and branch decisions; token PPO uses $r=\exp(\log\pi_{current}-\log\pi_{old})$ |
| PPO critic | Current/old values, returns, bootstrap masks and value clipping |
| GRPO | Complete groups, reward components, mean/std, normalization and epsilon |
| KL/entropy | Estimator, direction, coefficient and sampling distribution |

Use dependency or clipping diagnostics to inspect discrepancies.

## Framework integration

Select the native entrypoint and inspect the installed version's implementation.

| Framework / entrypoint | Inspect in the installed implementation |
|---|---|
| [TRL](https://huggingface.co/docs/trl/grpo_trainer) | Requested trainer's batch preparation, action scoring, loss and optimizer steps; retain completion/tool/truncation masks, group normalization, detached old scores and ordinary/fused loss selection |
| [verl](https://verl.readthedocs.io/en/latest/) | Sample/group/action mappings through repeats, filtering, balancing, packing and rank redistribution; capture effective old-policy anchors, correction/rejection weights, reward/KL placement and aggregation |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Experience construction, distributed worker roles, action masks, advantage estimator and actor/critic optimizer steps |
| [slime](https://github.com/THUDM/slime) | Rollout-data conversion, Megatron loss inputs and rollout/training weight-version mapping |
| [NeMo RL](https://docs.nvidia.com/nemo/rl/latest/index.html) | Selected policy/generation backends, batch layout, objective reduction and policy update |
| [ms-swift](https://github.com/modelscope/ms-swift) | Selected RLHF algorithm, template/masks, reward/group normalization and native trainer/backend |
| [AReaL](https://github.com/areal-project/AReaL) | Asynchronous trajectory/policy versions, action rescoring and effective off-policy corrections |
| [ROLL](https://github.com/alibaba/ROLL) | Role-specific pipeline, sample dispatch, reward/advantage processing and selected training backend |
| [MindSpeed-RL](https://github.com/Ascend/MindSpeed-RL) | Native actor/rollout/reward steps, sample redistribution and distributed parameter/gradient storage |

Record the RL framework, training backend and rollout engine separately. Preserve the policy/state versions used for each fixed trajectory. Use [training capture](sft.md#training-stages) for distributed tensors and the selected [inference entrypoint](inference.md#framework-integration) for rollout scores.

### Other/custom

Follow [native adaptation](../SKILL.md#adapt-an-unfamiliar-implementation). Locate native trajectory preparation, action scoring, objective evaluation and actor/critic updates. Preserve sample/group mappings and effective masks, then capture the objective state and training stages above.

## Metrics

| Observations | Report |
|---|---|
| Action scores | Logprob MAE, P99 and maximum at paired valid actions |
| Native objective | Each side's loss components and absolute differences; retain reward/advantage/value inputs for localization |
| Requested gradients, postclip gradients and updates | L2 error, RMSE, cosine and maximum absolute error |
| Coverage | Paired trajectories/groups/actions, effective masks, parameter/element counts and optimizer events |

Use [comparison](comparison.md) for score/tensor definitions and tool usage. Select the task adapter for native scores/objectives.
