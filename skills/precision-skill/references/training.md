# SFT and RL

## Common training state

Start from identical parameters and optimizer state. Align parameter/shard mapping, optimizer formula/hyperparameters, step/scheduler, master weights, batch/microstep order, accumulation, token normalization, scaling, communication and clipping. Record actual train mode and random quantities: matching seeds alone does not align cross-stack dropout or sampling.

After an update, states may diverge. Later steps are trajectory comparisons unless the common state is restored. Same-state numerical agreement and multi-step training effectiveness are separate questions.

## SFT

Compare one same-state step with GPU as reference. Collect **every valid sequence target token, full gradients and full actual parameter updates**.

| Stage | Collect | Compare |
|---|---|---|
| Forward/loss | Labels, loss mask, prediction/target positions, each valid target's logprob and loss term; objective components and denominators. | GPU/NPU loss, signed/absolute difference; target-logprob MAE, P99 and maximum error. |
| Backward | All elements of every present trainable-parameter gradient after accumulation, unscaling and communication, before clipping; gradient-presence inventory. | Relative L2, cosine, maximum absolute error; both norms and counts. |
| Update | Every trainable parameter before and after optimizer.step(), including unchanged elements; step/overflow/skipped-step records. | The same tensor metrics on actual deltas, computed separately on each side as after-before. |

Use the actual training loss mask: exclude padding/ignored labels, but include supervised EOS when active. The model still normalizes over the full vocabulary; comparison takes only the labeled token's score, not top-100 candidates. For plain cross-entropy, loss is minus the sum of target logprobs divided by the total valid-token count across sequences/ranks. Preserve the actual reduction and extra terms for other objectives.

If clipping is enabled, also collect its norm/coefficient and full post-clip gradients. Gradient absence must agree across sides and is not equivalent to a zero gradient. Final parameter closeness does not replace actual-update comparison.

Report tensor metrics globally over concatenated coordinates **and per logical parameter tensor**. Reconstruct shards, exclude padding, and avoid double-counting replicas/tied aliases. Stream by tensor/shard if needed; complete paired values are still required. Preserve stored before/after values, promote before subtraction to at least FP32, and accumulate metrics in FP64. Historical coordinate samples remain smoke tests, not full-coverage evidence. Metric formulas and undefined cases are in [comparison](comparison.md).

## RL

For TRL or verl, also read [RL frameworks](rl-frameworks.md) for collection boundaries; the metrics below are shared.

Use the actual algorithm and loss variant; do not infer PPO from the task label. Compare a fixed trajectory and one same-state update. Reuse the SFT full-gradient/actual-update collection rules above, including clipping stages, presence inventories, shard reconstruction and global/per-parameter metrics, separately for actor and critic when present.

Freeze prompt/response IDs, action positions and actual loss masks, EOS/truncation, policy identities and the external dependencies consumed by the update: behavior/old/reference action logprobs, rewards, advantages, returns or old values as applicable. Include active EOS and exclude masked actions. Behavior, old and reference policies are distinct roles; record their identities and score semantics, not assumed equality. Mark each dependency as injected or recomputed. Identical injected dependencies isolate downstream arithmetic; independently recomputed dependencies define a broader pipeline comparison and must not be described as identical update inputs.

Keep the default report to three metric families; retain supporting evidence without adding a metric dashboard:

| Stage | Collect | Default comparison |
|---|---|---|
| Actor scores | Current-policy logprob of every valid trajectory action, with history/token/position mapping and masks. | Action-logprob MAE, P99 and maximum error; token-weighted across complete paired coverage. |
| Objective | Actual total/component losses, coefficients, reduction denominators and dependencies needed to reconstruct the implemented objective. | Signed/absolute loss differences, separated by actor/critic; retain the algorithm's token/sequence/group reductions. |
| Backward/update | Full gradients and stored before/after trainable parameters, clipping and overflow/skipped-step evidence. | Existing relative L2, cosine and maximum absolute error for full gradients and actual updates. |

Retain algorithm-specific evidence only where used. PPO needs values/old values, returns and bootstrap/termination masks for a critic; clipped objectives need log-ratios/ratios, clip bounds and actual surrogate branch decisions. GRPO needs complete group membership, reward components, group statistics, normalization/epsilon and zero-variance handling. Preserve the implemented loss sign, weighting and token- versus sequence-level importance sampling. For token-level PPO-style ratios, r=exp(logpi_current-logpi_old); out-of-range ratios do not always select the clipped branch because advantage sign matters.

When a discrepancy warrants localization, inspect dependency/value errors, ratio errors or paired clipping-branch disagreement; these are conditional diagnostics, not additional default metrics. Different branch decisions can be a numerical finding under comparable inputs. Missing action/group coverage or incompatible objective/state semantics block the affected comparison; do not repair them by intersection, zero-fill or changing normalization.

Distinguish GPU/NPU distribution KL, current/old-policy approximate KL, and policy/reference KL. Record the implemented estimator, direction and sampling distribution when KL is used; sampled KL terms need not be nonnegative. Entropy is not a default precision metric; an objective requiring full entropy needs the full-distribution reduction, which top-100 cannot recover.

Selected actions may lie outside top-100, so collect their scores directly; inference distribution metrics do not replace RL action scoring. For rollout/trainer score comparisons, hold trajectory and policy identity fixed and identify pre-/post-sampling transforms; raw trainer scores and transformed behavior scores are not interchangeable. Reward correctness, online success rates and learning curves require separate upstream or task-effectiveness evaluation. No thresholds means numerical diagnostics, not an acceptance pass.
