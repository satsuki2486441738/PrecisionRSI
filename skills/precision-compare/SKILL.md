---
name: precision-compare
description: Collect and compare numerical results across hardware and software stacks for inference, SFT (full/LoRA) and RL.
---

# Precision comparison

## Scope

- Compare symmetric sides at matching inputs/state/coordinates; record intended differences and actual configurations. Ask when choices change task meaning or resource limits.
- Initially, collect or compare only missing scope. Route follow-ups by the requested action (default: reuse completed evidence):

| Request | Action |
|---|---|
| Repeat / confirm a completed task; inputs/state, source/runtime and scope unchanged | Return prior metrics, coverage and paths as reused; skip training/inference and regression reruns. |
| Conditions or requested scope changed | Recompute affected scope; retain unaffected checks. |
| Explicitly rerun training/inference or obtain new measurements | Execute fresh captures for the requested scope. |

## Tool use

Follow these requirements for the bundled tools:

| Operation | Required path |
|---|---|
| Interface details | Use declarations or `tool_help TOOL` / `--full`. Read tool source only for tool edits, observed tool failures or requested audits. |
| Supported work | Call the declared tool. Write adapters for uncovered semantics; identify an unsupported requirement before replacing a tool. |
| Results and checks | Reuse returned metrics, coverage and integrity receipts. Recheck only affected inputs/artifacts after changes, observed inconsistencies or requested audits. |
| Delivery | Return existing results, configurations and evidence paths. Create extra report files, bundles or offline replay only on request. |

## Compare

Reuse workload checks covering the request. Missing comparisons: import from `scripts/` (arrays/trees require NumPy):

| Input | Python call | Result |
|---|---|---|
| Named numeric arrays | `array_compare.compare_arrays(a, b, expected=None, metrics=False, groups=None)` | `exact_equal`, coverage, findings, optional groups |
| Nested state | `tree_compare.compare_trees(a, b, leaf_adapter=None, metrics=False)` | `exact_equal`, coverage, metadata, numeric |
| Coordinate → natural-logprob maps | `score_metrics.reduce_scores(a, b)` | Both NLLs, NLL difference, MAE/P99/max and coordinate |
| Global + grouped target scores | `group_metrics.reduce_groups(expected, groups, score_metrics.reduce_scores, a, b)` | `global` and `groups[name]`, each with status, coverage, metrics |

`expected`: task coordinate inventory; `groups`: label → coordinate subset. Arrays use their own `groups`. Adapters establish identities, masks, shifts and logical coordinates; tools supply arithmetic and coverage.

File CLIs, from the skill directory:

```sh
python scripts/array_compare.py A.npz B.npz --output NEW.json
python scripts/tree_compare.py A.pt B.pt --output NEW.json
python scripts/evidence_bundle.py fetch REQUEST.json NEW_SIDE_DIRECTORY
```

NPZ: named arrays. Checkpoints: trusted Torch state, loaded on CPU; Torch required. Both comparators accept `--metrics` and `--all-findings`. Exit 0=computed; equality is `exact_equal` (NPZ) or `comparison.exact_equal` (checkpoint).

Fetch request: `{"transport":["ssh","HOST","python3 -I -"],"source_dir":"/work/native-run","include":["*.json","*.bin"]}`. Use the endpoint's Python-stdin command and terminal receipt's source directory. Fetch verifies selected files. Reuse its receipt and `EXPORT.json` hashes/sizes for unchanged files; large arrays stay at source.

[tool_help](scripts/tool_help.py): `--list` for discovery, `--full` for schemas. Shards/runtime/repeated commands → [shared tools](references/comparison.md#tools).

## Collect

Execution tools own waiting, timeouts and terminal receipts. Use completion notifications when the host can resume the agent; otherwise use the longest supported blocking wait (at least 10 s when available). Keep scheduler polling inside execution adapters; inspect progress only to resolve a stall, handle interaction or answer a user request.

Independent sides: concurrent native capture → terminal receipts → batch-fetch selected small artifacts → compare. Use task recipes for unresolved capture semantics or framework integration.

- [Inference](references/inference.md): scores, distributions, cache stages.
- [SFT, full / LoRA](references/sft.md): objectives, gradients, stored updates.
- [RL](references/rl.md): trajectories, objectives, updates.

Preserve native operators, state transitions, masks and dtype identity; map batches/shards/replicas to logical coordinates. Missing execution → [adapt](../framework-adapt/SKILL.md). Confirm scope/resources before [internal sampling](references/layerwise.md).

## Return

- Return completed tool results, actual configurations, gaps and existing artifact paths. Add only uncovered task checks; large arrays stay at source/agreed storage.
- Recheck affected scope after changes, failed checks or unresolved findings.
- Apply supplied acceptance criteria; otherwise report differences. Requested [repair](../precision-repair/SKILL.md) reuses evidence; differences alone do not establish defects.
