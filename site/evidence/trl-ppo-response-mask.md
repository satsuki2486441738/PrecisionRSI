# TRL PPO response-mask historical backport

2026-10-22 · TRL 0.7.10 · native tiny CPU PPO workload. This reproduces an upstream-fixed defect and backports its runtime correction; it is independent of the IPO averaging repair.

## Cause and patch

Left padding shifts the response window in padded input coordinates. The original implementation also prepends query-sized zeros to a response-only mask and reslices it using those coordinates. A one-element slice then broadcasts over two response tokens, including an explicitly excluded token. The minimal patch removes this concatenation/reslicing and applies the response-only mask directly. Dtype, dependencies and PPO interfaces are unchanged.

The two-example fixture has input IDs `[[0,4,5,7,2],[4,5,6,8,2]]`, attention masks `[[0,1,1,1,1],[1,1,1,1,1]]` and response masks `[[0,1],[1,1]]`. In causal predictor coordinates the original mask is `[[0,0,1,1],[0,0,1,1]]`; the independent oracle expects `[[0,0,0,1],[0,0,1,1]]`.

## Independent oracle and native verification

The native value-head wrapper and PPOTrainer are normally constructed; the path covers `prepare_model_inputs`, `batched_forward_pass`, `loss`, backward and one direct SGD step (lr=0.01). A random GPT-2 has 1,121 parameters including the value head (one layer, hidden 8, two heads, vocabulary 12, seed 415, dropout zero). CPU, one thread, batch/minibatch two.

The oracle enumerates nonpadding token positions and maps selected response tokens to their preceding predictors, without TRL start/end slicing. An independent forward computes selected-token logsumexp scores and the scalar policy/value objective, then backward and SGD. The old policy/values match initial state: policy ratios are approximately one and value clipping is inactive. Clipping boundaries are outside this experiment.

| Left-padded FP32 error | Original | Patched |
|---|---:|---:|
| Mask mismatches / 8 coordinates | 1 | 0 |
| Selected tokens (oracle: 3) | 4 | 3 |
| Loss absolute error | 0.2708968458734202 | 3.594535114004316e-08 |
| Maximum parameter-gradient error / 1,121 | 0.5824297070503235 | 1.7881393432617188e-07 |
| Maximum stored SGD-update error / 1,121 | 0.005824297200888395 | 3.725290298461914e-09 |

Original regression: two failures, four passing controls; patched: six passes. FP32/FP64 each cover left padding with a response mask, right padding with a response mask, and left padding without an optional mask. All six before/after input and initial-state hashes, oracle coordinates and configurations match. All four controls are unchanged. Every case covers eight mask and 1,121 gradient/update coordinates; observations across cases are not independent models.

Masks require exact equality; loss/maximum gradient tolerance is 2e-6 and update tolerance 2e-7. The native wrapper emits FP32 logits even with FP64 parameters, so FP64 results are not a full FP64 scoring path. Residual rounding is retained in [structured evidence](trl-ppo-response-mask.json).

## Provenance and limits

- Affected TRL 0.7.10: `09ca7607d50714a80c9f0feb206b29f1c5a49bab`.
- Introduction: `9d09b3e107f1b1d0f31b0e144df68c382c504013`.
- Upstream fix: [`3bf94492a8dc84ac192f7c5206553e1460f53aa4`](https://github.com/huggingface/trl/commit/3bf94492a8dc84ac192f7c5206553e1460f53aa4), PR #1718, June 20, 2024. The upstream correction explicitly establishes response-only masks; the old misleading docstring is not the oracle.
- Checked upstream snapshot: `9c37460182946a5630da9b800a726f3bb27191f8`. Legacy PPOTrainer is absent from its public API and experimental tree; no current-API execution is claimed.

This verifies native CPU minibatch methods, backward and direct SGD, not `PPOTrainer.step` or a training loop. Generation, reward computation, GAE, clipping boundaries, optimizer scheduling, convergence, encoder-decoder, distributed and GPU/NPU workloads are untested. Timing is one cold sample per case, with no performance claim. Full execution artifacts remain with the private case; source/report/test hashes are retained in the JSON. Numerical acceptance awaits user judgment.

[Minimal runtime backport](trl-ppo-response-mask.patch) · [Measurements and provenance](trl-ppo-response-mask.json).
