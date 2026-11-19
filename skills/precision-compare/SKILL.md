---
name: precision-compare
description: Collect and compare numerical observations for inference, SFT and RL against an aligned reference or another configuration.
---

# Precision comparison

Start from the workload, reference, observation points and available evidence.

## Collect

Select exactly one recipe for the requested workload and read only that recipe:

- [Inference](references/inference.md): scores, distributions, cache stages.
- [SFT, full / LoRA](references/sft.md): objectives, gradients, stored updates.
- [RL](references/rl.md): trajectories, objectives, updates.

Read the [execution guide](references/comparison.md#wait-for-native-execution)
when you need shared receipts, workflow continuation or a tool interface. Read
[tool use](references/comparison.md#tools) only when selecting or integrating a
comparator. Read [internal sampling](references/layerwise.md) only when the
confirmed scope includes layerwise observations. Do not load unrelated recipes
or references. Run independent captures concurrently when resources and
isolation allow. Preserve native operators, state transitions and dtype identity.

## Tool use

Install with the repository exporter to bundle the tools. Use Python 3.10+ and
NumPy for array tools. Native execution uses the runtime required by the task.

Select a [comparator](references/comparison.md#tools) and use adapters for
task-specific coordinates, masks, shifts and state. Read [tool help](scripts/tool_help.py)
for missing interface details. Inspect source for edits, failures and audits.
Batch checks of retained results. Recompute affected results when their evidence
changes or a check calls them into doubt.
