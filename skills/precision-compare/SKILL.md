---
name: precision-compare
description: Collect and compare numerical results across hardware and software execution stacks for inference, SFT, LoRA, and RL. Use for fixed-input scoring, loss, gradient, or parameter-update comparisons across execution stacks.
---

# Precision comparison

Compare the requested observations at corresponding inputs, states and logical coordinates. Each execution has a side ID and actual configuration; comparison is symmetric. Establish task-selected differences and conditions that must match. Ask when a missing choice changes the comparison's meaning or resource limit.

Reuse existing captures when their identities, semantics and coverage match. For named arrays, [array_compare.py](scripts/array_compare.py) provides `compare_arrays(a, b, expected=names)` and an NPZ CLI. It returns exact equality, coverage and bounded findings; `metrics=True` adds error metrics. An independent expected inventory detects observations missing from both sides. Native loaders and explicit name mappings can supply arrays directly.

When new collection is needed, read the relevant recipe:

| Task | Collection and metrics |
|---|---|
| Inference | [Scores, distributions and cache stages](references/inference.md) |
| SFT (full / LoRA) | [Objectives, gradients and stored updates](references/sft.md) |
| RL | [Fixed trajectories, objectives and updates](references/rl.md) |

## Adapt an unfamiliar implementation

Capture at native boundaries, preserving operators, state transitions, masks and dtype identity. Map batches, shards and replicas to logical coordinates. Use [framework-adapt](../framework-adapt/SKILL.md) for missing execution support.

Use [artifact tools](references/comparison.md#tools) for shard reconstruction, execution-condition validation or remote queries; [shared task tools](references/comparison.md#shared-tasks) support repeated command chains. Write only missing integration.

Return requested results, actual configurations, paired/missing coverage, difference coordinates and reproduction artifacts. Label subsets and unavailable comparisons. Keep large arrays at source or the agreed store. Apply supplied acceptance requirements; otherwise report differences.

Before internal sampling, confirm scope/resources and follow [layerwise comparison](references/layerwise.md). For requested repairs, pass existing evidence to [precision-repair](../precision-repair/SKILL.md). A peer difference alone does not establish a defect.
