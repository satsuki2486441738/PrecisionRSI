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

Use the actual algorithm and loss variant; do not infer PPO from the task label. Compare a fixed trajectory and same-state update before drawing conclusions about online learning.

| Stage | Additional observations | Diagnostics |
|---|---|---|
| Trajectory/dependencies | Prompt/response IDs, action positions and masks, EOS/truncation, policy versions, sampling settings; behavior/old/reference action logprobs, rewards and advantages. | Exact common-input checks; action-logprob or independently produced dependency MAE/P99/max. |
| Actor objective | Current action logprobs, log-ratios/ratios, surrogate and KL terms, clipping decisions, coefficients and reduction denominators. | Component/total loss differences; ratio errors and range; clip fractions and paired clipping-decision disagreement. |
| Backward/update | Actor gradients and actual updates; critic equivalents when present. | Training tensor metrics, separated by model and stage. |
| PPO critic, when used | Values, old values, returns, bootstrap masks, value loss/clipping. | Value errors, loss differences and clipping disagreement. |
| GRPO groups, when used | Group IDs, reward components, group mean/std, advantage normalization/epsilon and zero-variance handling. | Dependency differences and group/normalization consistency. |

For token-level PPO-style objectives, r=exp(logpi_current-logpi_old). Preserve the implemented clipped surrogate, loss sign and normalization. Out-of-range ratios and active clipped branches differ with advantage sign; compare actual branch masks, not just aggregate fractions. Sequence-level importance sampling requires its own reduction.

Distinguish GPU/NPU distribution KL, current/old-policy approximate KL, and policy/reference KL. Record estimator, direction and sampling distribution; sampled KL terms are not necessarily nonnegative. Full entropy requires full-distribution reduction and cannot be recovered from top-100 alone.

Selected actions may lie outside top-100, so save their logprobs separately. Injected rewards/advantages establish downstream comparability, not correctness of their upstream producers. When comparing rollout and trainer scores, hold trajectory and policy version fixed and preserve sampling transforms. Report token-weighted diagnostics separately from algorithm-specific sequence/group reductions. Online rewards, success rates and learning curves require a separate task-effectiveness evaluation.
