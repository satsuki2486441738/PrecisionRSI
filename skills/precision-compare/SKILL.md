---
name: precision-compare
description: Collect and compare numerical results across hardware and software execution stacks for inference, SFT, LoRA, and RL. Use for fixed-input scoring, loss, gradient, or parameter-update comparisons across execution stacks.
---

# Precision comparison

Compare requested observations at matching inputs, states and logical coordinates. Each execution has a side ID and actual configuration; comparisons are symmetric. Establish intended differences and matching conditions. Ask when a missing choice changes meaning or resource limits.

Continue from known interfaces and retained evidence. Reuse observations matching identities, semantics and coverage; collect missing observations. Return retained reports for queries. Audit or rerun affected scope when verification, conditions or freshness require it.

## Collect

Use native capture interfaces. Select or extend boundaries and metrics with the relevant recipe:

| Task | Recipe |
|---|---|
| Inference | [Scores, distributions and cache stages](references/inference.md) |
| SFT (full / LoRA) | [Objectives, gradients and stored updates](references/sft.md) |
| RL | [Trajectories, objectives and updates](references/rl.md) |

Preserve native operators, state transitions, masks and dtype identity. Map batches, shards and replicas to logical coordinates. Use [framework-adapt](../framework-adapt/SKILL.md) for missing execution support. Confirm scope/resources before [internal sampling](references/layerwise.md).

## Compare and return

Use [compare_arrays](scripts/array_compare.py) for named observations, [reduce_groups](scripts/group_metrics.py) for requested groups, and [compare_trees](scripts/tree_compare.py) for nested state. Independent task inventories detect shared omissions. Native loaders and mappings belong to adapters. Query unknown interfaces together with [tool_help.py](scripts/tool_help.py): `python scripts/tool_help.py NAME ...`. Inspect implementation for an unresolved integration question. Read the [tool guide](references/comparison.md#tools) for shard reconstruction, runtime evidence checks or remote queries, and [shared tasks](references/comparison.md#shared-tasks) for repeated command chains.

Return requested metrics, actual configurations, coverage and reproduction paths; use bounded coordinate examples unless all coordinates are requested. Derive summaries from computed results; identify missing requested observations. Keep large arrays at source or the agreed store. Apply supplied acceptance requirements; otherwise report differences. For requested repairs, pass retained evidence to [precision-repair](../precision-repair/SKILL.md); a peer difference alone does not establish a defect.
