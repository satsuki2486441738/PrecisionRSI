---
name: precision-compare
description: Collect and compare numerical results across hardware and software execution stacks for inference, SFT, LoRA, and RL. Use for fixed-input scoring, loss, gradient, or parameter-update comparisons across execution stacks.
---

# Precision comparison

Compare requested observations at matching inputs, states and logical coordinates. Give executions side IDs and actual configurations; comparisons are symmetric. Establish intended differences; ask when missing choices change task meaning or resource limits.

Reuse retained observations matching identities, semantics and coverage. Return retained reports for queries; audit or rerun affected scope when verification, conditions or freshness require it.

## Compare and return

Start with supplied observations; collect missing ones. Use [compare_arrays](scripts/array_compare.py) for named observations, [reduce_groups](scripts/group_metrics.py) for requested groups, and [compare_trees](scripts/tree_compare.py) for nested state. For Torch checkpoint files: `python scripts/tree_compare.py A.pt B.pt --output NEW.json`. Other formats use adapters to load state and export framework leaves. Use returned coverage, counts and bounded findings directly; add traversal only for requested details absent from the report. Check an independent task inventory when available. Query needed interfaces together with [tool_help.py](scripts/tool_help.py): `python scripts/tool_help.py compare_trees compare_arrays`. Inspect implementation for an unresolved integration question. Read the [tool guide](references/comparison.md#tools) for shard reconstruction, runtime evidence checks or remote queries, and [shared tasks](references/comparison.md#shared-tasks) for repeated command chains.

Return requested metrics, configurations, coverage and reproduction paths; identify missing observations. Expand coordinate examples when all coordinates are requested. Keep large arrays at source or the agreed store. Apply supplied acceptance requirements; otherwise report differences. For requested repairs, pass retained evidence to [precision-repair](../precision-repair/SKILL.md); a peer difference alone does not establish a defect.

## Collect

Use native capture interfaces. Select or extend boundaries and metrics with the relevant recipe:

| Task | Recipe |
|---|---|
| Inference | [Scores, distributions and cache stages](references/inference.md) |
| SFT (full / LoRA) | [Objectives, gradients and stored updates](references/sft.md) |
| RL | [Trajectories, objectives and updates](references/rl.md) |

Preserve native operators, state transitions, masks and dtype identity. Map batches, shards and replicas to logical coordinates. Use [framework-adapt](../framework-adapt/SKILL.md) for missing execution support. Confirm scope/resources before [internal sampling](references/layerwise.md).
