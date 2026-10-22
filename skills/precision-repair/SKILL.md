---
name: precision-repair
description: Diagnose and fix numerical accuracy defects in model, framework, and operator implementations across execution environments. Use when asked to repair a numerical discrepancy or instability and verify the patch against the original workload.
---

# Precision repair

Reproduce a numerical problem, establish its cause, patch the responsible implementation and verify the original task. Start from a [precision-compare](../precision-compare/SKILL.md) report, a failing test or a workload description. Prepare environments and task adapters as needed; retain the task's model, data, algorithm and execution requirements.

## Establish the repair target

Identify the affected execution path, intended computation, correctness evidence, allowed changes, resource budget and acceptance requirements. Infer these from the request, artifacts, tests and source. Ask when a missing choice changes the intended result, permitted implementation or resource limit.

Use the algorithm definition, a trusted implementation, analytical properties or higher-precision recomputation to establish expected behavior. Check that the evidence implements the same masks, reductions, state transitions and boundary semantics. Differences between peer executions identify a question to investigate; determine which behavior is incorrect before choosing what to fix. When the remaining difference is an operator numerical or performance tradeoff rather than an incorrect computation, retain the isolated workload and measurements for a later optimization task. Record uncertainty when no available evidence resolves it.

Reuse the incoming contract, manifests, coverage and affected coordinates. Check artifact identities and replay retained comparisons when available; use native execution to reproduce the actual symptom. Retain the original inputs, state, source revisions, effective settings and comparison reports. Record reproduction and verification commands, outputs and exit status in fresh runs. For historical defects, identify the affected revision and any upstream fix; label a backport accordingly. Reproduce the symptom through the affected native entrypoint, recording actual devices, imported code and the selected implementation path. Estimate run-to-run variation when execution is nondeterministic. Select a small failing workload that preserves the relevant dtype, shape, layout, execution mode and distributed behavior.

## Locate and test the cause

Trace the affected output to corresponding internal boundaries. Reuse existing captures; when internal sampling is needed, obtain confirmation of its scope and resource limits unless already covered by the repair request. Within that scope, choose collectors and refine sampling autonomously.

Separate propagated input error from local computation error: replay the isolated operation with identical inputs and state, then compare its result with the correctness evidence. Inspect native dispatch, precision conversions, accumulation, reductions and state mutations as indicated by the observations. For training, include backward, accumulation and optimizer boundaries relevant to the failure. The [training replay example](examples/causal-replay.md) illustrates local error versus trajectory propagation.

Test a specific causal hypothesis with a bounded experiment. Preserve the initial failure and change one relevant factor at a time. Check that instrumentation and the experiment exercise the original implementation. Use the observed result to narrow the cause or revise the hypothesis.

## Patch and verify

Work in an isolated checkout of the responsible source. Add a regression that fails on the original behavior and checks the intended computation independently of the proposed fix. Patch the cause, preserving required interfaces and execution semantics. Record any intentional changes to numerical policy, dependencies or dispatch in the candidate configuration; verify that the modified code actually executes.

Exercise the failure and nearby valid cases selected by the cause, including dtype, shape, magnitude and layout where relevant. For differentiable or distributed paths, verify the affected backward and collective/state behavior. Reuse the task's acceptance requirements; justify any new local tolerance using the computation, dtype and correctness evidence.

Rerun the original task from the same inputs and initial state with the patch, using [precision-compare](../precision-compare/SKILL.md) collection and metrics. Compare both original and patched executions against the same correctness evidence, keeping coordinate selection, metrics and aggregation consistent. Report counts and uncovered regions alongside changes in error. Measure relevant runtime and memory costs under comparable conditions when a fix changes execution or precision policy.

Conclude against the requested acceptance requirements. If none were specified, report measured changes and remaining uncertainty for user review. If the cause or verification remains unresolved within budget, retain the reproducible workload, attempted changes and next discriminating experiment.

## Reuse tools and deliver

When installed alongside `precision-compare`, use its [tool index](../precision-compare/references/comparison.md#tools) for runtime checks, collection, artifact validation, numerical metrics and evidence packaging. For internal collection, read the relevant [layerwise integration](../precision-compare/references/layerwise.md). Resolve tools relative to the installed companion package or the task's checkout; load only the required interfaces. Without that package, use native tests and task-owned collection/comparison code for the same repair workflow.

Keep workload-specific collectors, experiments and patches with the case. Extract reusable operations into the existing tool library when their interfaces and regressions support multiple tasks.

Deliver the cause and supporting evidence, source revision and patch, regression results, original-task before/after metrics and coverage, actual configuration changes, measured costs and artifact paths. Preserve baseline artifacts and release allocated resources when the work ends.
