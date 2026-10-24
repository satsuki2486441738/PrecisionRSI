---
name: precision-repair
description: Diagnose and repair numerical defects in model, framework or operator implementations; verify the original workload.
---

# Precision repair

## Diagnose

- Enter for requested diagnosis or repair. Reuse the workload execution, inputs/state, source/runtime identities, collectors, captures and completed checks.
- Preserve model, data, algorithm and execution semantics. Ask when a choice changes intended results or permitted work.
- Use an existing native reproduction, or reproduce only the missing evidence through the affected path.
- Distinguish collection/configuration differences from defects. Reuse a trusted native reference/check; supply missing correctness evidence from the intended algorithm, an analytical property or matching higher-precision computation.
- Resolve causality with a small experiment; replay propagated training errors at identical inputs/state ([example](examples/causal-replay.md)). [Internal sampling](../precision-compare/references/layerwise.md) requires agreed scope/budget; reuse existing authorization.
- Report uncertainty when evidence cannot distinguish defects from numerical-policy tradeoffs.

## Patch and verify

- Patch the responsible implementation; keep framework-specific fixes with the case.
- Extend existing checks for missing causal or nearby-case coverage; the regression must fail before the fix and pass after.
- Verify through the original entrypoint and collectors at identical inputs/initial state. A regression invoking that workload can also provide its before/after metrics.
- Use task tolerance or justify one from computation/dtype. Measure runtime/memory when relevant.
- Existing captures: `compare_arrays` / `compare_trees`. Missing interface details: `python ../precision-compare/scripts/tool_help.py TOOL` from this skill. Missing collection/comparison → [compare](../precision-compare/SKILL.md); reconstruction/execution or repeated commands → [shared tools](../precision-compare/references/comparison.md#tools).

## Return

- Cause, source revision, patch/regression, original-workload before/after metrics, configuration changes and existing evidence paths.
- Finish after regression and original-workload checks pass; reuse results. Repeat only changed, failed or unresolved checks. Package, replay offline or write extra reports only when requested.
- If budget prevents verification, return the unresolved question and next discriminating check.
