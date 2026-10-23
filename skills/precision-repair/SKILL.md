---
name: precision-repair
description: Diagnose and fix numerical accuracy defects in model, framework, and operator implementations across execution environments. Use when asked to repair a numerical discrepancy or instability and verify the patch against the original workload.
---

# Precision repair

Fix the numerical problem and verify the original workload. Continue from its inputs/state, source/runtime identities, captures, reports and known interfaces. Preserve the requested model, data, algorithm and execution semantics; ask when a choice changes the intended result or permitted work.

## Establish the cause

Use a retained matching native reproduction or reproduce through the affected path. Distinguish collection/configuration differences from a defect using correctness evidence: intended algorithm, analytical property, trusted implementation or higher-precision recomputation with matching semantics. Report uncertainty when that evidence cannot distinguish a defect from a numerical-policy tradeoff.

Choose the smallest experiment that resolves the remaining causal question. For propagated training errors, compare the suspected computation at identical inputs/state; see the [training example](examples/causal-replay.md). Use [internal sampling](../precision-compare/references/layerwise.md) within the agreed scope and budget; the repair request may already establish them.

## Patch and verify

Patch the responsible implementation. Add a regression that fails before the fix and checks intended behavior independently of the patch; select nearby cases relevant to the cause.

Rerun the original workload from the same inputs and initial state using the same correctness evidence and metric definitions. One execution can establish both regression coverage and workload recovery. Use the task's tolerance, or justify a local tolerance from the computation and dtype. Measure runtime/memory when relevant to the requested result or changed numerical policy.

Use [compare_arrays](../precision-compare/scripts/array_compare.py) for corresponding captures and [artifact tools](../precision-compare/references/comparison.md#tools) for required reconstruction or execution validation. Continue existing task state; [shared task tools](../precision-compare/references/comparison.md#shared-tasks) can rerun affected command dependencies. Keep framework-specific fixes with the case.

Return the cause, source revision, patch/regression, original-workload before/after metrics, configuration changes and reproduction artifacts. Stop when the cause and recovery are verified; if budget prevents verification, return the unresolved question and next discriminating check.
