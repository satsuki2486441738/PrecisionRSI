---
name: framework-adapt
description: Establish native execution when a requested model, framework or hardware target needs deployment or backend adaptation.
---

# Framework adaptation

## Execute

Use [shared execution tools](../precision-compare/references/comparison.md#wait-for-native-execution) for waiting and receipts.

- Reuse environments, adapters and receipts matching source, dependencies and target.
- Missing support: inspect the selected revision's support declarations and closest native example; implement the required model/backend integration.
- Run or reuse a [native preflight](#native-preflight) before capture. Verify only the
  planned native path within the agreed scope and budget, then hand its receipt to
  comparison. A preflight is a readiness check and does not establish numerical
  agreement.
- Preserve native state transitions; record checkpoint conversions and tensor-layout changes.
- Keep implementations and environment declarations with the case. Use
  [repair verification](../precision-repair/SKILL.md#patch-and-verify) for patches.

## Native preflight

A preflight exercises the **smallest valid workload that reaches the declared
native path and produces its required evidence**. Choose the path and observable
completion checks before shrinking the workload. There is no fixed batch size,
sequence length or step count across tasks.

- **Keep the execution path.** Use the target device, model checkpoint and
  revision, framework and backend, dtype, and launch environment of the planned
  capture on that side. Preserve the topology and settings required to exercise
  the claimed path, including relevant kernels or collectives. A substitute
  model, CPU fallback, or a topology that skips a required collective is a
  separate check with narrower coverage.
- **Shrink work within that path.** Reduce requests, examples, sequence lengths
  or iterations only while the workload is valid and still reaches the required
  boundaries. Keep any shape or scale needed to trigger the path. Read only the
  recipe for the requested task: [inference](../precision-compare/references/inference.md),
  [SFT](../precision-compare/references/sft.md) or [RL](../precision-compare/references/rl.md#native-preflight).
- **Check execution and evidence.** Require successful native termination and
  the observed device and runtime path. Exercise the collectors needed for that
  path: verify representative records at each declared boundary against the
  expected keys, shapes and sample mapping. A file or successful launch alone
  does not establish capture coverage. Return these checks in the preflight
  receipt; gate formal capture on them. Reuse a matching receipt only for its
  recorded scope, and check the current device binding before new work.

## Continue

Hand the execution JSON, inputs/state and receipts to the next stage.

## Return

Return the runnable command, effective settings, readiness and gaps, and the
native receipt. In the actual worker, record device binding, imported framework
and backend versions/revisions, driver, runtime/toolkit (CANN or CUDA where
applicable) and the loaded collective library (HCCL or NCCL when used). Use
[the runtime probe](../precision-compare/scripts/runtime_probe.py) or an equivalent
native check; keep unconfirmed versions unknown.
