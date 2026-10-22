---
name: operator-optimize
description: Optimize operator numerical accuracy, latency or memory under explicit workload and correctness constraints. Use for kernel, fusion, accumulation or execution-policy improvements and verify their effect in the original inference, SFT or RL task.
---

# Operator optimization

Improve a selected operation against the requested numerical, latency or memory objective. Start from a workload, profiler trace, [precision-compare](../precision-compare/SKILL.md) evidence or a [precision-repair](../precision-repair/SKILL.md) investigation. Reuse existing environments, captures and task adapters; implement missing integration for the selected library and hardware.

## Define the objective and baseline

Identify the executed operation, workload shapes, dtypes, layouts, devices, modes and source/runtime revisions. Retain inputs, initial state, original-task commands, comparison contract and baseline reports. Determine the intended computation and the objective, numerical constraints and resource budget from the task. Ask when a missing choice determines an acceptable tradeoff or resource limit.

Establish correctness evidence using the algorithm, an independent implementation, analytical properties or higher-precision recomputation with matching masks, reductions and state semantics. Route an incorrect computation or state transition to precision-repair. Distinguish local operator error from error propagated through its inputs by replaying identical inputs and state.

Measure the original native operation and the original task before changing them. Select the task-relevant numerical metrics from [comparison](../precision-compare/references/comparison.md). For performance, record warmup, repetitions, synchronization, compilation inclusion, workload concurrency and measurement scope; report latency distributions and relevant peak memory using the platform's native measurement facilities.

## Implement and measure

Work in an isolated checkout. Select a change supported by the observed error or cost: accumulation, decomposition, fusion, layout, scheduling or dispatch. Preserve required computation, interfaces and state behavior; record changed numerical and execution policies. Verify that the intended implementation executes on the actual device.

Compare alternatives with identical local inputs and state. Exercise the target workload and nearby cases selected by the change, including relevant backward, collective and state behavior. Synchronize asynchronous execution at timing boundaries, separate compilation from steady-state timing, and account for run-to-run variation. Use regressions that independently check the intended computation.

## Verify the original task and deliver

Rerun the original inference, SFT or RL task through precision-compare with the same inputs, initial state, coordinates, metrics and aggregation. Retain the original baseline and save each alternative in a fresh run. Report numerical differences and coverage alongside latency, memory and their measurement scope. Evaluate any task-specified constraints; when constraints are absent, return the measured tradeoffs for user review.

Deliver the executed source revision and patch, correctness evidence, regression results, operator and original-task before/after measurements, actual configuration changes and artifact paths. Keep workload-specific benchmarks and patches with the case. Reuse the [comparison tool library](../precision-compare/references/comparison.md#tools) for runtime identity, run recording, validation and evidence packaging; extend shared tools when an operation is reusable and regression-tested. Release allocated resources when finished.
