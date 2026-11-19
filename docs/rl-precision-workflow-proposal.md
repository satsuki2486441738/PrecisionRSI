# First RL case design

Use this note to choose an RL research case and design its controls. Execute it
with the [NumericBridge workflow](../skills/numericbridge/references/workflow.md)
and the [RL comparison recipe](../skills/precision-compare/references/rl.md).

- **Case selection:** Prefer a real workload with a runnable environment,
  reusable evidence and an observable optimizer update. Name both the training
  backend and the rollout engine.
- **Comparison design:** Pair a short native rollout-to-update run with a fixed
  trajectory replay when needed to separate trajectory divergence from training
  numerics. Identify what the two paths share and which independent check could
  resolve a suspected defect.
- **Research scope:** Start with one concrete stack. Choose where to observe the
  update path and what question would justify adding another case, more updates
  or resume testing. Extract shared tooling when the case reveals a reusable gap.

Use the [community record interface](../community/records/README.md) to relate
published comparisons, findings, repair candidates and verifications.
