# SFT and RL comparison

## Same-state SFT step

Start both sides from the same parameters and optimizer state, with a trustworthy parameter/shard mapping. Freeze optimizer formula and hyperparameters, step, scheduler, master weights, loss scaling, batch/microstep order, accumulation, global token normalization, communication, and clipping stage.

Record the scalar objective and its definition, gradients at the agreed stage, and actual parameter changes. A completed `optimizer.step()` is not evidence of a nonzero update: save the required before/after values and compute their differences. Use the shared coordinate rules and relative-L2 definition in [comparison](comparison.md).

Record effective train/eval mode, dropout, sampling, and random operators. Matching seeds alone do not establish identical random quantities across stacks. Supply common random quantities or a controlled input when necessary; distinguish a controlled variant from the native path and state the remaining limits.

Freeze selection before collection when observing sampled parameters or shards. Save tensor identity, logical shape, global coordinates, counts, and shard mapping. Pair the complete agreed sampled set rather than silently keeping only its intersection. Conclusions apply to the saved coordinates; sparse observations do not establish full-gradient or full-update agreement.

A complete training-state comparison requires evidence of actual restoration of that state. Weight restoration or a small tensor readback establishes only its own scope. Save environment association at launch using [environment](environment.md), so later comparison can reuse either the runtime inventory or associated locked-environment records.

## RL dependencies

Choose the actual RL objective; do not infer PPO from the task label. Freeze prompt/response trajectories, valid tokens, termination/truncation rules, and the objective's old/reference logprobs, rewards, advantages, returns, and their provenance.

Check clipping, KL definition and weight, and token/sequence normalization. Common injected dependencies establish downstream comparability; their upstream production needs separate coverage. Distinguish fixed-trajectory numerical validation from online rollout behavior or task-quality claims.

## SFT

Anchored on 2026-10-15: compare one same-state SFT step with GPU as reference, every valid sequence target token, full gradients of all trainable parameters that have gradients, and actual updates of all trainable parameters including zero updates. Vocabulary top-100 comparison is outside this profile. Match the complete trainable-parameter and gradient-presence inventories across sides; an absent gradient is not a zero gradient. Thresholds remain null, so comparable results are diagnostics (`no_thresholds`), not acceptance passes.

The coverage and metrics below are agreed. Bind model, data, restored initial state, optimizer, numerical policies and device budget in an execution contract before collecting a new run. Existing first/last-element observations remain sampled smoke tests and cannot satisfy this profile. Current collectors/comparators need full-coverage support; this document does not assert implementation or authorize a device launch.

For each valid label y_t, collect log p(y_t | x_<t) from the actual train-mode forward at prediction position t-1. The model still normalizes over the full vocabulary; only the labeled token's score is collected for comparison. Preserve sequence identity, target position/ID and loss mask. Include every position selected by the training loss mask; exclude padding and ignored labels, and do not silently import inference's special-token exclusion (a supervised EOS is included if its loss mask is active). Reject missing/extra target coordinates rather than intersecting the two sides.

Compare paired target logprobs using MAE, global linear P99 and maximum absolute error. For plain cross-entropy, recompute loss as the negative sum of all valid target logprobs divided by their total count, and report GPU/NPU loss and signed/absolute difference. Aggregate across sequences and ranks by valid-token count, not an unweighted mean of sequence losses; per-sequence counts and diagnostics may be retained for localization. No vocabulary-distribution metrics are required by this SFT profile.

| Stage | Save | Compare or report |
|---|---|---|
| Replay and initial state | Input/position IDs, attention and loss masks, labels/shift, microbatch order and counts, parameter/optimizer/scheduler/scaler state identity and actual restoration evidence. | Exact semantic/coordinate checks; initial sampled values only certify sampled equality. |
| Objective | Every valid label's logprob and loss term; loss components, weights and reduction denominators; observed total objective. Include label smoothing, auxiliary losses and their dependencies when used. | Recompute token-weighted cross-entropy for the plain SFT case; signed and absolute loss difference; per-token logprob MAE/P99/max. Use the actual objective's reduction for other variants. |
| Backward | All elements of every present trainable-parameter gradient after accumulation, scaling removal and communication, before clipping; missing-gradient inventory. If clipping is enabled, record clipping norm/coefficient and full post-clip gradients too. | Relative L2, cosine, maximum absolute error; both sides' norms, finite counts, and full coordinate coverage. Do not equate missing gradients with zero gradients. |
| Optimizer update | All elements of every trainable parameter before and after, including unchanged elements; optimizer step counters, skipped-step/overflow flags, state dtypes and initial state; optional moments/master weights before and after. | Actual delta=after-before on each side; relative L2, cosine and maximum absolute error of deltas. Report changed-coordinate counts. Compare optimizer state separately if collected. |

An inference-only forward cannot stand in for the training forward. For a simple CE objective, PPL=exp(NLL) can be an auxiliary diagnostic; PPL is not exp(total loss) when other loss terms are included. Published SFT loggers also report entropy and token accuracy, as noted in the related-work table below; these are not required observations in the agreed target-token profile.

### Tensor metric rules

For reference r and candidate c, the required tensor metrics are relative L2 from [comparison](comparison.md), cosine=dot(r,c)/(norm(r)*norm(c)), and max absolute error. RMSE=sqrt(sum((c-r)^2)/N) is optional. Cosine is undefined if either norm is zero; report this explicitly without inventing a score or discarding the coordinate set. A perfectly equal pair of zero vectors has zero absolute error and relative L2 zero, but still undefined cosine. Scalar relative error is optional; a zero reference must be handled explicitly without epsilon.

Report tensor metrics both over the concatenation of all agreed coordinates and per parameter tensor (logical identity across shards). Global metrics use concatenated coordinates, not an unweighted mean of tensor ratios. Preserve per-tensor counts/norms and worst cases so large tensors do not conceal a local anomaly. A norm over sampled values is a sampled norm, not a full-model grad norm. If collecting a full gradient norm on device, avoid double-counting replicated shards and record the reduction stage.

Full coverage is mandatory for this profile: compare complete logical tensors across shards, verify element counts, exclude padding, and avoid double-counting replicas or tied aliases using an explicit mapping. Missing/extra coordinates or differing gradient-presence inventories block comparison; do not keep only an intersection or independently select large-magnitude elements.

Collection and comparison may stream by parameter or shard. Preserve lossless before/after values and matching gradients until paired validation completes; compute parameter deltas after promoting stored values to at least FP32, not by subtracting in BF16. Accumulate comparison norms/dot products/errors in FP64 with stable reductions and record the implementation. Single-side norms cannot replace paired values. A retained summary may replace bulk intermediate files only after coverage and metrics have been validated and the case's retention policy is satisfied.

## Recommended RL collection and metrics

Separate three scopes: rollout forward, a fixed-trajectory actor/critic step, and online learning effectiveness. Start with fixed-trajectory numerical comparison; the actual algorithm determines required fields. A PPO/GRPO logger is a source of diagnostic ideas, not a universal RL loss definition.

| Stage | Save | Compare or report |
|---|---|---|
| Frozen trajectory | Prompt/response IDs, action positions, attention/response/loss masks, EOS/truncation, sample/group IDs, policy versions and sampling configuration. | Exact histories, action coordinates and valid masks; top-100 rollout distribution metrics where selected. After independent rollouts diverge, positional pairing is invalid. |
| Objective dependencies | Behavior/old-policy and reference-policy logprobs for **every selected action**, reward components and weights, advantages before/after normalization; returns, values and bootstrap masks if used. Record which values are common injected inputs and which are produced on each side. | Exact equality for injected inputs; MAE/P99/max and signed mean difference for separately produced numeric dependencies. Reward/advantage production requires separate coverage if injected. |
| Actor objective | Current selected-action logprobs, log-ratios/ratios, per-token or per-sequence surrogate terms, clipping decisions, KL estimator terms, masks and all component denominators. | Selected-action logprob MAE/P99/max; ratio MAE/P99/max and per-side min/mean/max; signed/absolute differences of total and component losses; clipping fractions and **paired clipping-decision disagreement rate**. |
| Entropy and KL | Actual entropy computation and KL formula, direction, reference policy, sampling policy, estimator and coefficient. | Per-side values and cross-device differences. Separate old/current-policy approximate KL from policy/reference KL and GPU/NPU distribution KL. Full entropy cannot be recovered from top-100 alone. |
| Backward/update | Actor gradients and before/after parameters; critic equivalents when present; optimizer/scaler/communication records as for SFT. | Same gradient and actual-update tensor metrics as SFT, reported separately for actor and critic. |
| PPO critic, if present | Predicted and old values, returns, advantages, terminal/bootstrap semantics, value clipping and loss terms. | Value/return errors, value loss difference and clipping disagreement; optional explained variance with an explicit undefined case for zero return variance. GRPO without a critic does not require these fields. |

For token-level PPO-style objectives, r_t=exp(logpi_current(a_t)-logpi_old(a_t)); preserve both log-ratios and ratios. Sequence-level importance sampling has a different reduction and must be recorded. The usual clipped surrogate is min(r_t*A_t, clip(r_t,1-eps_low,1+eps_high)*A_t), with the loss sign and reduction frozen. Save both the out-of-range mask and the active clipped-branch mask: they differ depending on advantage sign. Freeze tie handling. Compare masks at the same coordinates; matching scalar clipping fractions can hide different clipped tokens.

Record any train/rollout importance correction separately, including behavior logprobs, temperature and sampling transforms. Measure selected-action logprob mismatch between sampler and trainer on the same frozen trajectory and policy version; preserve mean absolute/P99/max errors. Do not conflate this within-stack mismatch with GPU/NPU disagreement. Do not change an algorithm's estimator or objective to obtain nicer numerical agreement.

For GRPO, preserve complete prompt groups, reward components, group mean/std, normalization formula/epsilon, zero-variance handling and final advantages. Variants differ in clipping and token/sequence length normalization; record the exact loss variant. Finite trajectory averages of sampled KL terms are not necessarily the exact full-vocabulary KL or nonnegative.

Aggregate action errors over the frozen valid action set. Report token-weighted diagnostics separately from algorithm-specific token/sequence/group-weighted objectives. Logprob or ratio overflow/nonfinite values are explicit failures, not removable outliers. For online effectiveness, separately agree on task reward/success rate, evaluation data, seeds and uncertainty, completion lengths and termination rates; one fixed step cannot establish learning quality.

## Evidence from related work

Sources reviewed on 2026-10-15; upstream development documentation can change. Pin implementation revisions in executable cases.

| Source | What it actually records or recommends | Use here |
|---|---|---|
| [TRL SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer#logged-metrics) | Cross-entropy loss, entropy, mean token accuracy, pre-clipping gradient norm. | Training observability; these scalars alone do not establish gradient/update agreement. |
| [msprobe training precision guide](https://github.com/Ascend/msprobe/blob/master/docs/zh/best_practices/train_debug_guide.md) | Separates convergence from numerical issues; inspects loss, grad norms, forward/backward tensors, optimizer states, and RL train/rollout logprob mismatch. | Common state and stage-specific comparisons; no blanket threshold imported. |
| [msprobe PyTorch comparison](https://github.com/Ascend/msprobe/blob/master/docs/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md) | Cosine, Euclidean distance, absolute/relative errors and elementwise tolerance ratios. | Tensor diagnostic precedent. Our relative-L2/update protocol is an explicit design choice, not a claimed universal tool default. |
| [TRL PPOTrainer v0.15.2](https://github.com/huggingface/trl/blob/v0.15.2/docs/source/ppo_trainer.md#explanation-of-the-logged-metrics) | Policy/reference KL, old/current approximate KL, entropy, rewards, policy/value losses, ratio and clip fractions. | Conditional PPO diagnostics; preserve the distinction between its different KL metrics. |
| [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer#logged-metrics) | Reward mean/std, zero-variance groups, entropy, reference KL when enabled, clipping and sampler/trainer mismatch metrics. | Conditional group and sampling diagnostics; not a mandate to use GRPO. |

Cross-device paired errors, actual-update comparisons and clipping-mask disagreement are our recommended additions to those training logs. No cited training metric supplies a universal GPU/NPU acceptance threshold. Existing case comparators may implement only a subset; mark uncollected recommendations as uncovered rather than claiming they were checked.

## State and trajectory divergence

| Mode | Required interpretation |
|---|---|
| Same-state numerical comparison | The observation point has common inputs, initial state, objective dependencies, and actual random-condition evidence. |
| Free trajectory or training effectiveness | States/history may evolve; separately freeze task, dataset split, generation settings, and outcome metrics. |
| Same-device version regression | Fix the device and other conditions; identify the reference version and intended change. |

After the first update, parameters and optimizer states can diverge. Later steps compare trajectories unless a common state is restored for each same-state step. After generation/rollout divergence, retain both histories and stop interpreting different tokens at equal indices as common-input observations.

Numerical consistency and task effectiveness answer separate questions. Scoring evidence alone does not establish backward or update execution.

## Collector and comparison readiness

Before device execution, specify the objective, input replay, effective modes, normalization, gradient stage, optimizer initial state, coordinate mapping, saved observations, completion checks, and environment association. Accept a minimal collection before expanding it.

The bundled schema-v2 score comparator implements full-forward logprobs. Prepare case-level training collectors and validators for losses, gradients, and actual updates; preserve their format/version and evidence with the case. Report actual coverage of objective, backward, and update separately. Use a frozen threshold when acceptance is requested; otherwise report numerical diagnostics.
