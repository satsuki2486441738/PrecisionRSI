---
name: precision-repair
description: Diagnose and repair numerical defects in model, framework or operator implementations; verify the original workload.
---

# Precision repair

## Diagnose

- Reuse matching inputs/state, source/runtime identities, captures, reports and interfaces.
- Preserve model, data, algorithm and execution semantics. Ask when a choice changes intended results or permitted work.
- Reuse a matching native reproduction, or reproduce through the affected path.
- Distinguish collection/configuration differences from defects. Correctness evidence: intended algorithm, analytical property, trusted implementation or higher-precision recomputation with matching semantics.
- Resolve causality with a small experiment; replay propagated training errors at identical inputs/state ([example](examples/causal-replay.md)). [Internal sampling](../precision-compare/references/layerwise.md) requires agreed scope/budget; reuse existing authorization.
- Report uncertainty when evidence cannot distinguish defects from numerical-policy tradeoffs.

## Patch and verify

- Patch the responsible implementation; keep framework-specific fixes with the case.
- Regression: fail before the fix; independently check intended behavior and relevant nearby cases.
- Rerun the original workload with identical inputs/initial state, correctness evidence and metrics. Combine regression/recovery runs when possible.
- Use task tolerance or justify one from computation/dtype. Measure runtime/memory when relevant.
- Corresponding captures → [array comparison](../precision-compare/scripts/array_compare.py); reconstruction/execution validation → [artifact tools](../precision-compare/references/comparison.md#tools); dependency reruns → [shared tasks](../precision-compare/references/comparison.md#shared-tasks).

## Return

- Cause, source revision, patch/regression, original-workload before/after metrics, configuration changes and existing evidence paths.
- Finish after regression and original-workload checks pass; reuse results. Repeat only changed, failed or unresolved checks. Package, replay offline or write extra reports only when requested.
- If budget prevents verification, return the unresolved question and next discriminating check.
