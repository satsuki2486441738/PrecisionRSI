---
name: precision-repair
description: Diagnose and repair numerical defects in model, framework or operator implementations; verify the original workload.
---

# Precision repair

Start from the confirmed request, paired observations and evidence paths.

## Diagnose

1. **Reproduce the observation.** Reuse the native small case, identities,
   collectors and captures. Check that inputs, initial state, effective settings
   and observed coordinates are comparable; resolve capture or configuration
   mismatches first. Repeat only when needed to distinguish a stable discrepancy
   from run-to-run variation. Preserve the intended execution semantics.
2. **Localize the difference.** Start at the collected stage boundaries and
   add [internal observations](../precision-compare/references/layerwise.md)
   only around the suspect interval within the agreed scope and budget. Report
   the earliest **observed** divergent boundary and the nearest checked upstream
   boundary that agreed, if one exists. Unobserved or unpaired boundaries leave
   the true first divergence open.
3. **Test the cause.** Inspect the responsible code. If upstream state has
   diverged, [replay with identical inputs and state](examples/causal-replay.md)
   to test whether the suspect stage introduces or propagates the difference.
   Change one factor at a time when explanations compete. Judge correctness
   using the intended algorithm, an invariant or an aligned independent
   reference; a difference between stacks alone does not decide which is wrong.

```xml
<diagnosis_handoff>
  Update the case's local issues.md with the discrepancy, paired coordinates,
  coverage and checked boundaries, including gaps. Cite supporting evidence
  and the source revision and code when identified. Classify the discrepancy
  as a capture or setup problem, a numerical-policy difference, a supported
  defect, or unresolved.
  Propose a patch only when correctness evidence supports the defect. Pass
  a focused regression check to verification, or the smallest discriminating
  next check when the cause remains open.
</diagnosis_handoff>
```

## Patch and verify

- Test each patch, upgrade or settings change separately against the same verified
  baseline. Keep framework-specific patches with the case.
- On the original small native case, show the focused check fail before and pass
  after. Reuse matching baseline evidence, entrypoint, collectors, inputs, initial
  state and reference. Change only the candidate under test.
- Reuse those runs for workload verification when coverage suffices. Extend only
  to cover the trigger, affected behavior or agreed delivery scope.
- Apply the agreed tolerance, or justify a regression tolerance from the computation
  and dtype. Measure runtime and memory costs when relevant.

Use [precision-compare](../precision-compare/SKILL.md) for missing captures or
metrics and [shared execution](../precision-compare/references/comparison.md#shared-tasks)
for receipts or resumed runs.

## Return

Update the same issue entry after each candidate attempt with its change and
tested configuration, regression and workload outcomes, before/after metrics
when measured, coverage and evidence paths. Mark verified only after both
checks pass; retain failed candidates and remaining checks for partial results.
Return the issue record path with the diagnosis and candidate outcomes.
