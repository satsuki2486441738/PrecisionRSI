---
name: precision-compare
description: Collect and compare numerical results across hardware and software execution stacks for inference, SFT, LoRA, and RL. Use for fixed-input scoring, loss, gradient, or parameter-update comparisons across execution stacks.
---

# Precision comparison

Collect and compare numerical observations at corresponding inputs, states and coordinates. Each execution has a side ID and actual hardware/software settings; comparison is symmetric.

Identify the differences under study and conditions that must match: model/weights, inputs, algorithm, numerical policy and relevant state. Reuse existing environments, adapters and captures when their identities and coverage match. Use bounded sampling defaults; ask when a missing choice changes the comparison's meaning or resource limit.

Read only the relevant collection recipe:

| Task | Collection and metrics |
|---|---|
| Inference | [Scores, distributions and cache stages](references/inference.md) |
| SFT (full / LoRA) | [Objectives, gradients and stored updates](references/sft.md) |
| RL | [Fixed trajectories, objectives and updates](references/rl.md) |

## Adapt an unfamiliar implementation

Integrate at the native capture boundary, preserving operators and state transitions. Map batching, shards and replicas to logical coordinates. Record native and derived dtypes, computation-affecting capture options and the actual executing runtime/device. Use [framework-adapt](../framework-adapt/SKILL.md) for missing execution support.

Select the needed [comparison tools](references/comparison.md#tools) by interface; write missing task adapters. Existing artifacts can be compared directly. Use [shared task tools](references/comparison.md#shared-tasks) for repeated command chains and continuation.

Before aggregation, check coordinate pairing, masks and expected coverage. Return actual configurations, metric values/definitions, coverage, maximum-error coordinates and reproduction artifacts. Label subsets and unavailable observations; keep large arrays at source or the agreed artifact store.

Apply user acceptance requirements when supplied; otherwise report measured differences. Before extending model-level comparison into internal sampling, confirm scope and resources, then follow [layerwise comparison](references/layerwise.md). When repair is requested, continue with [precision-repair](../precision-repair/SKILL.md) using the existing evidence. A peer difference alone does not establish a defect.
