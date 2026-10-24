---
name: framework-adapt
description: Establish native execution when a requested model, framework or hardware target needs deployment or backend adaptation.
---

# Framework adaptation

## Execute

Execution tools own waiting, timeouts and terminal receipts. Use completion notifications when the host can resume the agent; otherwise use the longest supported blocking wait (at least 10 s when available). Keep scheduler polling inside execution adapters; inspect progress only to resolve a stall, handle interaction or answer a user request.

- Enter for missing native execution. Reuse environments, adapters and receipts matching source, dependencies and target.
- Missing support: inspect the selected revision's support declarations and closest native example; implement the required model/backend integration.
- Reuse a matching successful preflight; otherwise check a minimal native call with the capture configuration. Verify the original workload; retry within budget.
- Preserve native state transitions; record checkpoint conversions and tensor-layout changes. Confirm changes to target or implementation scope.
- Patches need a focused regression and original-workload verification. Keep implementations and environment declarations with the case.

## Continue

When requested work needs [comparison](../precision-compare/SKILL.md) or [repair](../precision-repair/SKILL.md), continue with the same execution JSON (`argv_prefix`, `env`, `path_prepend`), inputs/state and evidence. Use [shared task tools](../precision-compare/references/comparison.md#shared-tasks) directly for repeated command chains.

## Return

Return support/gaps, effective settings, device execution, source/dependency identities, runnable command and existing evidence paths once workload checks pass. Recheck only changed or unresolved scope; packaging and extra reports require a request.
