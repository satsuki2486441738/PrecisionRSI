# SFT and RL

## Common training state

Compare one step from identical parameters and optimizer state. Align shard mapping, optimizer formula/hyperparameters, master weights, step/scheduler, microbatch order, accumulation, scaling, communication and clipping. Preserve train mode and actual random quantities; equal seeds alone do not align dropout. Later steps compare trajectories unless the common state is restored.

| Stage | Required observations | Comparison |
|---|---|---|
| Objective | Effective scores/masks, total/component losses, coefficients, per-microstep denominators and backward scaling. | Signed/absolute loss differences using the actual token/sequence/group and distributed reduction, not averaged rank logs. |
| Backward | Every present trainable-parameter gradient after accumulation, unscale and communication, before clipping; presence inventory. When clipping is enabled, also its norm/coefficient and all post-clip gradients. | Relative L2, cosine, maximum absolute error, globally and per logical parameter. |
| Update | All stored trainable parameters immediately before/after the step, including unchanged elements; overflow/skipped-step events. | The same metrics for $\Delta\theta=\theta_{\mathrm{after}}-\theta_{\mathrm{before}}$. |

Reconstruct complete logical tensors, exclude shard padding and deduplicate replicas/tied aliases. Gradient absence is not zero. Preserve stored values, promote to at least FP32 before subtraction, and compute metrics in FP64; stream by shard/tensor if needed. Coordinate samples do not establish full coverage. See [comparison](comparison.md) for formulas and undefined cases.

A numerically skipped step is an event: retain finite comparable scores/losses and stored updates, label skipped versus applied, and block only affected invalid scopes. Do not describe it as agreement between two applied updates.

## SFT

Collect every valid target token's label, prediction/target position, logprob and loss term using the actual loss mask. Exclude ignored/padded positions; include active EOS. Scores remain full-vocabulary-normalized, but only the labeled token is compared.

Report target-logprob MAE, P99 and maximum error, plus the objective/gradient/update metrics above. Plain CE uses the global valid-token mean of negative logprobs; preserve the implemented reduction and extra terms for other objectives.

## RL

Use the actual algorithm/loss variant and fixed trajectories. For TRL or verl, read [RL frameworks](rl-frameworks.md) for collection boundaries. Apply the common training rules separately to actor and critic when present.

Freeze prompt/response IDs, action positions, base masks and EOS/truncation. Record policy identities and effective dependencies: behavior/old/reference scores, rewards, advantages, returns or old values as used. Mark each as injected or recomputed. Common injected values isolate downstream arithmetic; recomputed values extend the measured pipeline. Preserve detach semantics, sampling transforms and dynamic objective masks.

Keep three default metric families: current-action logprob MAE/P99/max, objective differences, and full-gradient/actual-update errors. Action errors are token-weighted across all valid actions, including active EOS; collect their scores even outside top-100. Objective reductions remain algorithm-specific.

Retain evidence for the selected algorithm:

- Clipped objectives: log-ratios/ratios, bounds and actual branch decisions. For token-level PPO, $r=\exp(\log\pi_{\mathrm{current}}-\log\pi_{\mathrm{old}})$ at the same action/history; advantage sign determines whether the clipped branch is active. Preserve sequence-level reductions when used.
- PPO critic: current/old values, returns, bootstrap/termination masks and value-loss clipping.
- GRPO: complete groups, reward components, mean/std, normalization/epsilon and zero-variance handling.
- KL/entropy terms: estimator, direction and sampling distribution. GPU/NPU distribution KL, current/old and policy/reference KL are different quantities; sampled estimates may be negative. Full entropy cannot be recovered from top-100.

Dependency/value errors, ratio errors and clipping disagreement are conditional localization diagnostics. Numerical branch changes may be findings; missing actions/groups or incompatible objective semantics block the affected scope. Rollout/trainer scoring requires the same policy/history and compatible transforms. Reward correctness and online learning effectiveness are separate evaluations.
