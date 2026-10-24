---
name: precision-repair
description: Diagnose and repair numerical defects in model, framework or operator implementations; verify the original workload.
---

# Precision repair

## Diagnose

- Enter for requested diagnosis or repair. Reuse the workload execution, inputs/state, source/runtime identities, collectors, captures and completed checks.
- Preserve model, data, algorithm and execution semantics. Ask when a choice changes intended results or permitted work.
- Use an existing native reproduction, or reproduce only the missing evidence through the affected path.
- Distinguish collection/configuration differences from defects. Correctness evidence: intended algorithm, analytical property, trusted implementation or higher-precision recomputation with matching semantics.
- Resolve causality with a small experiment; replay propagated training errors at identical inputs/state ([example](examples/causal-replay.md)). [Internal sampling](../precision-compare/references/layerwise.md) requires agreed scope/budget; reuse existing authorization.
- Report uncertainty when evidence cannot distinguish defects from numerical-policy tradeoffs.

## Patch and verify

- Patch the responsible implementation; keep framework-specific fixes with the case.
- Regression: fail before the fix; independently check intended behavior and relevant nearby cases.
- Verify the patch through the existing workload entrypoint and collectors, with identical inputs/initial state, correctness evidence and metrics. Combine regression/recovery runs when possible.
- Use task tolerance or justify one from computation/dtype. Measure runtime/memory when relevant.
- Existing captures → [array tool](../precision-compare/scripts/array_compare.py); reconstruction/execution checks → [artifact tools](../precision-compare/references/comparison.md#tools); missing collection/comparison → [compare](../precision-compare/SKILL.md); repeated commands → [shared tasks](../precision-compare/references/comparison.md#shared-tasks).

## Return

- Cause, source revision, patch/regression, original-workload before/after metrics, configuration changes and existing evidence paths.
- Finish after regression and original-workload checks pass; reuse results. Repeat only changed, failed or unresolved checks. Package, replay offline or write extra reports only when requested.
- If budget prevents verification, return the unresolved question and next discriminating check.
