---
name: precision-skill
description: Collect and compare numerical results across hardware and software execution stacks for inference, SFT, LoRA, and RL. Use for fixed-input scoring, loss, gradient, or parameter-update comparisons across execution stacks.
---

# Precision comparison

Complete the requested collection and comparison, including environment preparation, native framework integration and task-specific code. Select the execution setups and pairs to compare. Each side has an ID; hardware, runtime and framework are its configuration. For existing captures, check model/input/state identities and requested coordinates, then use [comparison](references/comparison.md).

## Collect

Determine the common conditions and allowed differences from the task and environment inspection before collection. Check model/weight/tokenizer/config identities, input IDs/positions/masks, algorithm and loss reductions, numerical settings, capture boundaries and metric definitions. For training, also check initial parameters, optimizer state, microbatch order, accumulation, clipping and random inputs. Record conditions that must match separately from each side's actual configuration, including requested differences and implementation differences required by its execution stack. Validate execution against these settings; ask about an unexpected difference when its effect on the comparison's meaning is unclear. Pin dependencies per setup and retain native operators.

Infer missing settings from the task, existing configuration and environment. Use small, bounded defaults for an initial run and record them. Ask when missing information determines the comparison's meaning or resource limit and cannot be inferred from context.

Reuse available environments, resources and collectors; prepare missing dependencies and implement missing collection or comparison code. Resolve connections, native entrypoint, device capacity, allocation, output path and timeout from the task. Read the selected recipe:

| Task | Recipe |
|---|---|
| Inference | [Scores, cache stages and serving engines](references/inference.md) |
| SFT (full / LoRA) | [Training targets, state and updates](references/sft.md) |
| RL | [Fixed trajectories, objectives and updates](references/rl.md) |

Run the requested stages within the task's cumulative budget. Save coordinates, arrays, effective settings, imported package/runtime versions and per-rank completion together. Validate at the source; compare there or transfer large arrays through the agreed artifact store with size/hash verification. Return small reports locally. Release resources and account for cumulative usage before retrying; fix collection/launch errors within the budget.

## Use and extend tools

`scripts/` contains callable tools and their supporting modules. Select an existing tool from the [tool index](references/comparison.md#tools) by its inputs, operation and outputs. Use CLI tools through `--help` and Python tools through their documented functions; read the selected module docstring for its artifact interface. Load implementation code only when integration or adaptation requires it.

Task adapters prepare native inputs, execute the framework and map observations to logical coordinates. Tools handle reusable planning, export, validation and comparison; third-party packages such as msProbe provide optional backends. Keep task-specific code with the task. When an existing tool does not cover the requested operation, implement the missing adapter or tool; during skill development, extract reusable code with a documented input/output interface and meaningful regression coverage.

## Adapt an unfamiliar implementation

Use the installed version's documentation and source to locate batch preparation, native execution and capture boundaries. Execute through its native entrypoint, operators and state transitions. Map observations to the task's logical coordinates, preserving native storage dtypes and distinguishing captured from derived quantities.

Map collected observations to the [comparison interfaces](references/comparison.md). Report unavailable requested observations with their affected coverage.

## Compare and report

Use the task's metric selection and [comparison tools and definitions](references/comparison.md). Return actual settings, metric values and aggregations, coverage counts, maximum-error coordinates and artifact paths.

Before switching from model-level comparison to internal layerwise collection, ask the user to confirm the transition, scope and resource limits. After confirmation, autonomously select and integrate collectors, verify sampling and refine within the approved scope and budget using [layerwise comparison](references/layerwise.md), including msProbe capabilities and integration steps.
