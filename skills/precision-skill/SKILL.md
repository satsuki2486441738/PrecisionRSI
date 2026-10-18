---
name: precision-skill
description: Collect and compare numerical results across hardware and software execution stacks for inference, SFT, LoRA, and RL. Use for fixed-input scoring, loss, gradient, or parameter-update comparisons across execution stacks.
---

# Precision comparison

Select the execution setups and pairs to compare. Each side has an ID; hardware, runtime and framework are its configuration. For existing captures, check model/input/state identities and requested coordinates, then use [comparison](references/comparison.md).

## Collect

Match model/checkpoint/tokenizer, input IDs/positions/masks, algorithm and reductions, numerical policy and replay state. For training, include optimizer groups, moments/step, microbatch order, accumulation, clipping and actual random inputs. Pin compatible dependencies independently per setup; align upstream versions when a shared-implementation comparison is requested. Retain native operators.

Reuse available environments, resources and collectors. Resolve connections, native entrypoint, device capacity, allocation, output path and timeout from the task. Read the selected recipe:

| Task | Recipe |
|---|---|
| Inference | [Scores, cache stages and serving engines](references/inference.md) |
| SFT (full / LoRA) | [Training targets, state and updates](references/sft.md) |
| RL | [Fixed trajectories, objectives and updates](references/rl.md) |

Run the requested stages within the task's cumulative budget. Save coordinates, arrays, effective settings, imported package/runtime versions and per-rank completion together. Validate at the source; compare there or transfer large arrays through the agreed artifact store with size/hash verification. Return small reports locally. Release resources and account for cumulative usage before retrying; fix collection/launch errors within the budget.

## Adapt an unfamiliar implementation

Use the installed version's documentation and source to locate batch preparation, native execution and capture boundaries. Execute through its native entrypoint, operators and state transitions. Map observations to the task's logical coordinates, preserving native storage dtypes and distinguishing captured from derived quantities.

Keep collector adapters with the task and map observations to the [comparison interfaces](references/comparison.md). Report unavailable requested observations with their affected coverage. Extract reusable framework guidance when repeated use warrants it.

## Compare and report

Use the task's metric selection and [comparison tools and definitions](references/comparison.md). Return settings used, principal differences, compared coordinate counts, largest discrepancies and output paths. Evaluate supplied tolerances against the computed metrics.
