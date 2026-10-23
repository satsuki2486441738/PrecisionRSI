---
name: precision-compare
description: Collect and compare numerical results across hardware and software stacks for inference, SFT (full/LoRA) and RL.
---

# Precision comparison

## Scope

- Compare symmetric sides at matching inputs/state/coordinates; record intended differences and actual configurations. Ask when choices change task meaning or resource limits.
- Reuse observations matching identities, semantics and coverage. Return retained reports for queries; audit/rerun affected scope for verification, changed conditions or freshness.

## Compare

Start with supplied observations; collect missing ones. Use these declarations directly:

| Input | Python call | Result |
|---|---|---|
| Named numeric arrays | `array_compare.compare_arrays(a, b, expected=None, metrics=False)` | `exact_equal`, `coverage`, `findings` |
| Nested state | `tree_compare.compare_trees(a, b, leaf_adapter=None, metrics=False)` | `exact_equal`, `coverage`, `metadata`, `numeric` |
| Coordinate groups | `group_metrics.reduce_groups(expected, groups, reducer, *observations)` | `global`, `groups` |

Python imports use `scripts/` on `sys.path` (NumPy required). File CLIs, from the skill directory:

```sh
python scripts/array_compare.py A.npz B.npz --output NEW.json
python scripts/tree_compare.py A.pt B.pt --output NEW.json
```

NPZ: named arrays. Checkpoints: trusted Torch state, loaded on CPU; Torch required. Both accept `--metrics`. Exit 0=computed; equality is `exact_equal` (NPZ) or `comparison.exact_equal` (checkpoint).

Reuse report coverage/findings. Query [tool_help](scripts/tool_help.py) for missing details (`--full` for schemas); `--list` discovers tools. Read source for unresolved integration issues/tool changes. Shards/runtime/remote operations → [tools](references/comparison.md#tools); repeated commands → [shared tasks](references/comparison.md#shared-tasks).

## Collect

Use native capture; load the relevant recipe.

- [Inference](references/inference.md): scores, distributions, cache stages.
- [SFT, full / LoRA](references/sft.md): objectives, gradients, stored updates.
- [RL](references/rl.md): trajectories, objectives, updates.

Preserve native operators, state transitions, masks and dtype identity; map batches/shards/replicas to logical coordinates. Missing execution → [adapt](../framework-adapt/SKILL.md). Confirm scope/resources before [internal sampling](references/layerwise.md).

## Return

- Metrics, actual configurations, coverage/gaps and reproduction paths; all coordinates when requested. Large arrays stay at source/agreed storage.
- Apply supplied acceptance criteria; otherwise report differences. Requested [repair](../precision-repair/SKILL.md) reuses evidence; differences alone do not establish defects.
