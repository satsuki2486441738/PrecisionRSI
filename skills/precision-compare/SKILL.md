---
name: precision-compare
description: Collect and compare numerical results across hardware and software stacks for inference, SFT (full/LoRA) and RL.
---

# Precision comparison

## Scope

- Compare symmetric sides at matching inputs/state/coordinates; record intended differences and actual configurations. Ask when choices change task meaning or resource limits.
- Enter for missing collection or comparison. Reuse workload checks, observations and results matching identities, semantics and coverage. Refresh changed scope or requested fresh measurements.

## Compare

Use these tools directly for missing comparisons. Native preflight/capture can reuse execution JSON via [shared tasks](references/comparison.md#shared-tasks):

| Input | Python call | Result |
|---|---|---|
| Named numeric arrays | `array_compare.compare_arrays(a, b, expected=None, metrics=False, groups=None)` | `exact_equal`, `coverage`, `findings`, optional groups |
| Nested state | `tree_compare.compare_trees(a, b, leaf_adapter=None, metrics=False)` | `exact_equal`, `coverage`, `metadata`, `numeric` |
| Coordinate → natural-logprob maps | `score_metrics.reduce_scores(a, b)` | NLL, NLL difference, MAE/P99/max and coordinate |
| Coordinate groups | `group_metrics.reduce_groups(expected, groups, reducer, *observations)` | `global`, `groups` |
| Remote artifacts | `evidence_bundle.fetch(transport, source_dir, include, output)` | Verified local files, hashes, byte counts |

Python imports use `scripts/` on `sys.path`; array/tree comparisons require NumPy. File CLIs, from the skill directory:

```sh
python scripts/array_compare.py A.npz B.npz --output NEW.json
python scripts/tree_compare.py A.pt B.pt --output NEW.json
```

NPZ: named arrays. Checkpoints: trusted Torch state, loaded on CPU; Torch required. Both accept `--metrics`; `--all-findings` returns every differing coordinate and native value. Checkpoint paths use original state keys/indices. Exit 0=computed; equality is `exact_equal` (NPZ) or `comparison.exact_equal` (checkpoint).

Use existing checks when they cover the request; otherwise use shared comparators and `groups` for requested stage summaries. Full configuration/identity files are code inputs; reuse report coverage/findings. Query [tool_help](scripts/tool_help.py) for missing details (`--full` for schemas); `--list` discovers tools. Read source for unresolved integration issues/tool changes. Shards/runtime/remote operations → [tools](references/comparison.md#tools); repeated commands → [shared tasks](references/comparison.md#shared-tasks).

## Collect

Independent sides: concurrent native capture → terminal receipts → batch-fetch selected small artifacts → compare. Use task recipes for unresolved capture semantics or framework integration.

- [Inference](references/inference.md): scores, distributions, cache stages.
- [SFT, full / LoRA](references/sft.md): objectives, gradients, stored updates.
- [RL](references/rl.md): trajectories, objectives, updates.

Preserve native operators, state transitions, masks and dtype identity; map batches/shards/replicas to logical coordinates. Missing execution → [adapt](../framework-adapt/SKILL.md). Confirm scope/resources before [internal sampling](references/layerwise.md).

## Return

- Return completed tool results, actual configurations, gaps and existing artifact paths. Add only uncovered task checks; recompute tool arithmetic for requested audits or suspected tool defects. Large arrays stay at source/agreed storage.
- Recheck affected scope after changes, failed checks or unresolved findings. Package, replay offline or write extra reports only when requested.
- Apply supplied acceptance criteria; otherwise report differences. Requested [repair](../precision-repair/SKILL.md) reuses evidence; differences alone do not establish defects.
