---
name: framework-adapt
description: Deploy or adapt an ML execution framework for a specified model, hardware and workload. Use for integrating a new framework or extending model/backend support, producing a reproducible native execution path and validation evidence.
---

# Framework adaptation

Make the requested framework execute the model workload. Deliver its reusable environment, invocation and required adapter or patch.

Continue from retained receipts and known interfaces. Reuse environments and adapters when their source, dependencies and target match. For missing support, check the selected revision's support declarations and closest native example, then identify the model/backend capability to implement. Confirm changes to the requested target or implementation scope.

Validate missing support with the smallest native workload that exercises it, then run the requested workload. Retain actual device execution, imported package/source identities and effective settings. Preserve native state transitions; make checkpoint conversion and tensor-layout transformations explicit.

Use retained failures to choose the next change within the resource budget. For implementation changes, add a focused regression and verify the original workload. Keep framework-specific code and reusable environment declarations with the case.

For numerical validation, continue with [precision-compare](../precision-compare/SKILL.md); for a confirmed correctness defect requiring repair, use [precision-repair](../precision-repair/SKILL.md). Pass existing inputs, state and artifacts between components. For repeated command chains, use the [shared task tools](../precision-compare/references/comparison.md#shared-tasks).

Return achieved support, unresolved gaps, environment/source identities, runnable command and validation artifacts. Existing receipts can supply this evidence.
