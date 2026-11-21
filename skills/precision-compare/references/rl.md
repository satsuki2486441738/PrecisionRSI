# RL

## Native preflight

Apply the [native preflight contract](../../framework-adapt/SKILL.md#native-preflight)
to the RL path in scope. Find the minimum valid sample count, complete group
size, microbatch and accumulation window in the selected configuration. Use
prompts and responses that yield valid, unmasked actions and keep the workers,
topology and settings needed by the path. A sample count of one can bypass the
intended computation or be invalid, for example for group-based advantages.

| Requested path | Evidence required in the reduced run |
|---|---|
| Rollout only | Native generation returns valid actions and the requested scores or other observations. |
| Training update | The native trainer consumes valid trajectories, evaluates the in-scope objective, runs backward and completes an optimizer update without skipping it. Captured state identifies the update. Fixed trajectories are valid for this scope; label injected dependencies. |
| Online rollout-to-update loop | Run native rollout and the configured reward/advantage path into a training update. Verify that the next native rollout consumed the resulting actor policy version. A transfer success message alone does not establish use. |

For example, use one valid training update and the next rollout for the online
preflight, with enough trajectories to satisfy the objective and distributed
batching. This checks one handoff; more iterations or the target workload are
needed for claims about repeated updates, long sequences or scale.

## Collection

For the first full-loop comparison, start with a fixed prompt set sized for a
valid update, one optimizer update, and the first rollout using the updated
policy. Keep evidence of policy consumption separate from evidence of numerical
agreement. Pair per-action scores only at matching histories and distinguish
common model state from independently updated state at the same step number.

For each in-scope stage, verify the common initial model and optimizer state
where applicable, then pair the required scoring, reward/advantage, loss,
selected gradients and actual parameter deltas. When both native stacks are in
scope and online divergence prevents those pairs, replay the smallest valid
fixed trajectory batch on both stacks from the matched state. Reuse existing
paired captures and replay only enough to cover the missing stage. Mark the
aligned comparison incomplete if a required native capture is unavailable;
CPU objective recomputation or a score-only probe covers only its own stages.

For each claimed group-centered reward gradient check, retain mixed-group and
valid-action counts after filtering and masking on both sides, plus a nonzero
reward-driven reference gradient before cross-prompt aggregation. Put coverage
in the comparison receipt. An optimizer step or nonzero total gradient cannot
satisfy this check. If coverage is missing, reuse matching evidence or run the
smallest [fixed mixed-reward replay](../examples/minimal-precision-batch.md#fixed-precision-batch)
that covers it. Keep online and replay results separate; do not rerun the online
batch just to fill this gap.

For replay, fix prompt/response IDs, action positions, masks and EOS/truncation. For objective comparisons, retain the effective old/reference scores, rewards, advantages, returns and values used by that objective, marking injected versus recomputed dependencies. Preserve native score transforms, detach behavior and dynamic masks.

For training comparisons, apply the requested [training state and stages](sft.md#training-stages) to actor and critic; identify the optimizer step, accumulation window and actor/critic order. Use the relevant objective dependencies below:

| Objective component | State to capture |
|---|---|
| Clipped policy objective | Ratios, bounds and branch decisions; token PPO uses $r=\exp(\log\pi_{current}-\log\pi_{old})$ |
| PPO critic | Current/old values, returns, bootstrap masks and value clipping |
| GRPO | Complete groups, reward components, mean/std, normalization and epsilon |
| KL/entropy | Estimator, direction, coefficient and sampling distribution |

## Framework integration

Use the relevant entrypoint below. Inspect the installed implementation when trajectory, objective or capture semantics are unresolved.

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

Follow [native adaptation](../../framework-adapt/SKILL.md#execute). Locate native trajectory preparation, action scoring, objective evaluation and actor/critic updates. Preserve sample/group mappings and effective masks, then capture the objective state and training stages above.

## Metrics

For exact-equivalence checks, report equality, coverage and mismatch coordinates. Quantify other differences with the relevant metrics below.

| Observations | Available measures |
|---|---|
| Action scores | Logprob MAE, P99 and maximum at paired valid actions |
| Native objective | Each side's loss components and absolute differences; retain reward/advantage/value inputs for localization |
| Requested gradients, postclip gradients and updates | L2 error, RMSE, cosine and maximum absolute error |
| Coverage | Paired trajectories/groups/actions, effective masks, parameter/element counts and optimizer events |

Use [comparison](comparison.md) for score/tensor definitions and tool usage. Select the task adapter for native scores/objectives.
