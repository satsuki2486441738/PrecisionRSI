---
name: precision-compare
description: Collect and compare numerical results across hardware and software stacks for inference, SFT (full/LoRA) and RL.
---

# Precision comparison

## Scope

- Assign side IDs; compare symmetrically at matching inputs/state and logical coordinates.
- Establish intended differences and actual configurations. Ask when missing choices change task meaning or resource limits.
- Reuse observations matching identities, semantics and coverage. Return retained reports for queries; audit/rerun affected scope for verification, changed conditions or freshness.

## Compare

Start with supplied observations; collect missing ones. Select by input:

| Input | Tool |
|---|---|
| Named arrays | [compare_arrays](scripts/array_compare.py) |
| Requested groups | [reduce_groups](scripts/group_metrics.py) |
| Nested state | [compare_trees](scripts/tree_compare.py) |
| Torch checkpoints | `python scripts/tree_compare.py A.pt B.pt --output NEW.json` |

Other formats: loading/leaf adapters. Reuse coverage/counts/findings; traverse for missing requested details. Check any independent task inventory.

Query selected interfaces together: `python scripts/tool_help.py compare_trees compare_arrays` ([help](scripts/tool_help.py)). Read implementation for unresolved integration questions. Shards/runtime evidence/remote queries → [tools](references/comparison.md#tools); repeated commands → [shared tasks](references/comparison.md#shared-tasks).

## Collect

Use native capture; load the relevant recipe.

- [Inference](references/inference.md): scores, distributions, cache stages.
- [SFT, full / LoRA](references/sft.md): objectives, gradients, stored updates.
- [RL](references/rl.md): trajectories, objectives, updates.

Preserve native operators, state transitions, masks and dtype identity; map batches/shards/replicas to logical coordinates. Missing execution → [adapt](../framework-adapt/SKILL.md). Confirm scope/resources before [internal sampling](references/layerwise.md).

## Return

- Metrics, actual configurations, coverage/gaps and reproduction paths; all coordinates when requested. Large arrays stay at source/agreed storage.
- Apply supplied acceptance criteria; otherwise report differences. Requested [repair](../precision-repair/SKILL.md) reuses evidence; differences alone do not establish defects.
