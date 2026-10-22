---
name: precision-compare
description: Collect and compare numerical results across hardware and software execution stacks for inference, SFT, LoRA, and RL. Use for fixed-input scoring, loss, gradient, or parameter-update comparisons across execution stacks.
---

# Precision comparison

Complete the requested collection and comparison, including environment preparation, native framework integration and task-specific code. Select the execution setups and pairs to compare. Each side has an ID; hardware, runtime and framework are its configuration. For existing captures, check model/input/state identities and requested coordinates, then use [comparison](references/comparison.md).

## Collect

Determine the common conditions and allowed differences from the task and environment inspection before collection. Check model/weight/tokenizer/config identities, input IDs/positions/masks, algorithm and loss reductions, numerical settings, capture boundaries and metric definitions. For training, also check initial parameters, optimizer state, microbatch order, accumulation, clipping and random inputs. Record conditions that must match separately from each side's actual configuration, including requested differences and implementation differences required by its execution stack. Validate execution against these settings; ask about an unexpected difference when its effect on the comparison's meaning is unclear. Select an officially supported dependency combination for each setup, align versions as required by the task, and retain native operators.

Infer missing settings from the task, existing configuration and environment. Use small, bounded defaults for an initial run and record them. Ask when missing information determines the comparison's meaning or resource limit and cannot be inferred from context.

Reuse available environments, resources and collectors; prepare missing dependencies and implement missing collection or comparison code. Use the [runtime probe](scripts/runtime_probe.py) to check ordered initialization and selected interfaces in the chosen interpreter, then run a minimal task through native execution, capture, export, validation and comparison before expanding collection. Record imported module paths/versions separately from distribution metadata. Resolve connections, native entrypoint, device capacity, allocation, output path and timeout from the task. Record actual execution devices inside native workers and associate them with resource allocations using platform device identities or topology. Read the selected recipe:

| Task | Recipe |
|---|---|
| Inference | [Scores, cache stages and serving engines](references/inference.md) |
| SFT (full / LoRA) | [Training targets, state and updates](references/sft.md) |
| RL | [Fixed trajectories, objectives and updates](references/rl.md) |

Run the requested stages within the task's cumulative budget. Save coordinates, arrays, effective settings, imported package/runtime versions and per-rank completion together. Validate at the source; compare there or transfer large arrays through the agreed artifact store with size/hash verification. Return small reports locally. Keep full job logs with the run and read compact status at stage changes. Release resources and account for cumulative usage before retrying; fix collection/launch errors within the budget. Once requested artifacts validate and metrics are available, release remaining resources and deliver the report.

## Use and extend tools

`scripts/` contains callable tools and their supporting modules. Select an existing tool from the [tool index](references/comparison.md#tools) by its inputs, operation and outputs. Use CLI tools through `--help` and Python tools through their documented functions; read the selected module docstring for its artifact interface. Load implementation code only when integration or adaptation requires it.

Task adapters prepare native inputs, execute the framework and map observations to logical coordinates. Tools provide composable planning, export, validation and comparison operations; third-party packages such as msProbe are optional backends. Implement missing adapters or tools as needed. During skill development, extract reusable operations with explicit interfaces and regression coverage for correctness and supported input variations; keep task-specific settings in adapters and environments.

## Adapt an unfamiliar implementation

Use the installed version's documentation and source to locate batch preparation, native execution and capture boundaries. Execute through its native entrypoint, operators and state transitions. Map observations to the task's logical coordinates, preserving native storage dtypes and distinguishing captured from derived quantities.

Map collected observations to the [comparison interfaces](references/comparison.md). Report unavailable requested observations with their affected coverage.

## Compare and report

Use the task's metric selection and [comparison tools and definitions](references/comparison.md). Return actual settings, metric values and aggregations, coverage counts, maximum-error coordinates and artifact paths.

Before switching from model-level comparison to internal layerwise collection, ask the user to confirm the transition, scope and resource limits. After confirmation, autonomously select and integrate collectors, verify sampling and refine within the approved scope and budget using [layerwise comparison](references/layerwise.md), including msProbe capabilities and integration steps.

## Continue into repair or optimization

When repair is requested, pass the comparison contract, per-side manifests and actual configurations, reports with coverage and affected coordinates, and the original inputs/state or their verified artifact locations to [precision-repair](../precision-repair/SKILL.md). Include source/runtime identities and the native collection commands so the symptom can be rerun. First resolve invalid captures, mismatched conditions and adapter errors; a peer difference alone does not establish a library defect.

For an operator accuracy, latency or memory objective, pass the same workload and numerical evidence to [operator-optimize](../operator-optimize/SKILL.md), with the requested objective and constraints. Preserve baseline artifacts and reuse the same comparison definitions for every changed implementation. The default comparison task ends with metrics and coverage; continue when the user's task includes repair or optimization.
