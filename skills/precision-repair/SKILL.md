---
name: precision-repair
description: Diagnose reproducible failures of supported native paths and numerical defects; verify fixes on the original workload.
---

# Precision repair

## Diagnose

1. **Reproduce.** Reuse the native trigger and evidence. For a preflight failure,
   confirm support on the tested revision, then retain the failing command,
   configuration and earliest failing operation. For a numerical difference,
   check input/state, settings and coordinate alignment before replaying it.
   Repeat only to resolve reproducibility or variation.
2. **Localize.** Start with captured boundaries; add [internal observations](../precision-compare/references/layerwise.md)
   only around the suspect interval. Report the earliest observed failure or
   divergence and the closest checked upstream boundary that passed, if known.
3. **Test the cause.** Inspect the relevant source and change one factor at a
   time. For diverged upstream state, [replay with identical inputs and state](examples/causal-replay.md)
   to distinguish introduced from propagated error. Judge numerical correctness
   against the intended algorithm, an invariant or an independent reference.

```xml
<diagnosis_handoff>
  In issues.md, classify the issue as setup/capture, unsupported capability,
  expected numerical policy, supported defect or unresolved. Link source and
  evidence, checked boundaries and ruled-out explanations. Continue beyond
  localization while a focused check can advance the cause within scope and
  budget. If blocked, leave the concrete blocker and next check. For a supported
  defect, hand off the trigger and focused regression to authorized repair.
</diagnosis_handoff>
```

## Patch and verify

- Test one patch, upgrade or configuration change at a time. Keep case-specific
  patches with the case and reuse matching before evidence, inputs and settings.
- Show the focused regression fail before and pass after, then verify the
  original small native case. For an execution defect, verify the original
  failing path and its required evidence; report numerical coverage separately.
  Reuse the regression run when it covers both checks. Extend only for missing
  coverage or agreed scope.
- Use the agreed numerical tolerance or justify the regression tolerance from
  the computation and dtype. Measure runtime and memory when relevant.

Use [precision-compare](../precision-compare/SKILL.md#collect) for missing captures
or metrics, and [shared execution](../precision-compare/references/comparison.md#recover-an-interrupted-stage)
for resumed runs.

## Return

Update the same issues.md entry with each candidate, tested configuration,
before/after observations, regression and workload outcomes, evidence and
remaining checks. Mark verified only after both checks pass.
