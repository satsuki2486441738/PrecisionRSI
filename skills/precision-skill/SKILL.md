---
name: precision-skill
description: Collect and compare numerical results across hardware and software execution stacks for inference, SFT, LoRA, and RL. Use for fixed-input scoring, loss, gradient, or parameter-update comparisons across execution stacks.
---

# Precision comparison

Complete the requested collection and comparison, including environment preparation, native framework integration and task-specific code. Select the execution setups and pairs to compare. Each side has an ID; hardware, runtime and framework are its configuration. For existing captures, check model/input/state identities and requested coordinates, then use [comparison](references/comparison.md).

## Collect

Determine the common conditions and allowed differences from the task and environment inspection before collection. Check model/weight/tokenizer/config identities, input IDs/positions/masks, algorithm and loss reductions, numerical settings, capture boundaries and metric definitions. For training, also check initial parameters, optimizer state, microbatch order, accumulation, clipping and random inputs. Record conditions that must match separately from each side's actual configuration, including requested differences and implementation differences required by its execution stack. Validate execution against these settings; ask about an unexpected difference when its effect on the comparison's meaning is unclear. Pin dependencies per setup and retain native operators.

Reuse available environments, resources and collectors; prepare missing dependencies and implement missing collection or comparison code. Resolve connections, native entrypoint, device capacity, allocation, output path and timeout from the task. Read the selected recipe:

| Task | Recipe |
|---|---|
| Inference | [Scores, cache stages and serving engines](references/inference.md) |
| SFT (full / LoRA) | [Training targets, state and updates](references/sft.md) |
| RL | [Fixed trajectories, objectives and updates](references/rl.md) |

Run the requested stages within the task's cumulative budget. Save coordinates, arrays, effective settings, imported package/runtime versions and per-rank completion together. Validate at the source; compare there or transfer large arrays through the agreed artifact store with size/hash verification. Return small reports locally. Release resources and account for cumulative usage before retrying; fix collection/launch errors within the budget.

## Adapt an unfamiliar implementation

Use the installed version's documentation and source to locate batch preparation, native execution and capture boundaries. Execute through its native entrypoint, operators and state transitions. Map observations to the task's logical coordinates, preserving native storage dtypes and distinguishing captured from derived quantities.

Keep collector adapters with the task and map observations to the [comparison interfaces](references/comparison.md). Report unavailable requested observations with their affected coverage. During skill development, extract reusable task code into the tool library and keep task-specific logic with the task.

## Compare and report

Use the task's metric selection and [comparison tools and definitions](references/comparison.md). Return actual settings, metric values and aggregations, coverage counts, maximum-error coordinates and artifact paths.
