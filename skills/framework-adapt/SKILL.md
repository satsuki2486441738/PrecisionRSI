---
name: framework-adapt
description: Deploy or adapt an ML framework for a specified model, hardware and workload; verify native execution.
---

# Framework adaptation

## Execute

- Reuse environments, adapters and receipts matching source, dependencies and target.
- Missing support: inspect the selected revision's support declarations and closest native example; implement the required model/backend integration.
- Validate the missing capability on a small native workload, then the original workload; retry within budget.
- Preserve native state transitions; record checkpoint conversions and tensor-layout changes. Confirm changes to target or implementation scope.
- Patches need a focused regression and original-workload verification. Keep implementations and environment declarations with the case.

## Continue

Pass existing inputs/state and evidence to [comparison](../precision-compare/SKILL.md) or requested [repair](../precision-repair/SKILL.md). Repeated command chains can use [shared task tools](../precision-compare/references/comparison.md#shared-tasks).

## Return

Support/gaps, effective settings, device execution, imported source/dependency identities, runnable command and validation artifacts.
