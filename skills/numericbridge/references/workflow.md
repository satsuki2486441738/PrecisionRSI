# NumericBridge workflow

## Stages and handoffs

Use the [shared execution tools](../../precision-compare/references/comparison.md#shared-tasks) for plans, receipts and recovery.

```xml
<handoff>
  Keep the confirmed request, plan, workspace, runtime and workload identities,
  native job/process IDs, evidence, issue record, remaining budget and next action.
  Put known dependent commands and required output checks in one plan. Plan
  completion covers only its declared checks. Continue authorized work without
  pausing at a progress update; advance independent work when one path blocks.
  In a batch, prioritize actionable open issues over new cases within budget.
</handoff>

<issue_record>
  Record findings that affect execution or results in the case's issues.md when
  found, including setup blockers. Update the same entry after diagnosis and
  every candidate: affected configuration, current conclusion, verified scope,
  open checks and linked attempts, including failures. Distinguish workarounds
  from fixes on the original path.
</issue_record>
```

## Request and execution inputs

Start from the [confirmed request](request.md). Carry its authorization forward;
resolve routine setup and retries within scope. Before native launch, resolve
placement, host or queue, launch method, allocation and budget, checkpoint and
revision, runtime and settings. Ask only for missing required inputs the agent
cannot resolve or new authorization.

## Route community results

```xml
<entry>
  The target and task are confirmed.
</entry>

<action>
  Call search_community with hardware, model, library and task (inference,
  sft or rl). Pass version only when pinned and model_size when needed.
  Correct invalid inputs; use the same query with CLI fallback if unavailable.
</action>

<output>
  Record lookup status, source and index identity. Carry matching issues,
  recommendations and solutions with their tested settings and scope.
  Distinguish no match from unavailable lookup.
</output>

<next>
  Prepare the environment even without a match. Preserve pinned versions.
</next>
```

Use the [bundled search CLI](../../precision-compare/scripts/community_search.py)
when MCP or its index is unavailable (`--model-size` for `model_size`).

## Prepare and compare

### Prepare the environment

Use [framework-adapt](../../framework-adapt/SKILL.md#execute) and its
[native preflight contract](../../framework-adapt/SKILL.md#native-preflight).

```xml
<entry>
  Community lookup finished or was unavailable.
</entry>

<action>
  Prepare target and reference independently. Reuse matching environments and
  receipts. Confirm device binding before launch and preflight each unverified
  native path and its required collectors.
</action>

<output>
  Record side labels, effective execution JSON, runtime and resource receipts,
  workload, readiness and any uncovered paths.
</output>

<next>
  Compare ready paths. For a failed preflight, first recover the original run
  if interrupted. Then distinguish setup or resource problems, unsupported
  capabilities, and reproducible failures of a supported path. Correct setup
  within scope or report the blocker. Route supported-path failures to diagnosis
  with a native reproducer, even without comparison metrics. If the reference
  remains blocked, use the fallback below when applicable.
</next>
```

### Run the comparison

Use [precision-compare](../../precision-compare/SKILL.md#collect) and its task
recipe. Without a specified dataset, use the
[fixed 100-example batch](../../precision-compare/examples/precision-batch-v1.json)
in order on both sides; see the [task views](../../precision-compare/examples/minimal-precision-batch.md).

```xml
<entry>
  Native capture is ready on the paths under test. If no usable reference is
  available, run only standalone checks and state their limits.
</entry>

<action>
  Align inputs, initial state and coordinates; declare intended differences.
  For online RL, allow native sampling and pair only matching histories.
  Follow the task recipe for grouping, sampling, updates and fixed replay.
</action>

<output>
  Record workload and side identities, expected and paired observations by
  stage, gaps, metrics with coordinates and evidence. Identify derived dtypes.
</output>

<next>
  For critical missing or unpairable observations, recover the needed setup or
  capture; unmatched observations are coverage gaps. Unless limited to comparison,
  route tolerance violations, reproducible output flips, clear error amplification
  after updates, or failed independent reference or invariant checks to diagnosis.
  Hand off the trigger, coordinates, observed values and evidence. Otherwise deliver.
</next>
```

### Choose a baseline and reference

```xml
<reference>
  For cross-platform work, prefer executable access to both sides. With only
  the target accelerator, independently recompute suitable checkpoints on CPU
  (FP64 where appropriate) and report their coverage. Judge accuracy against
  an analytic result, validated independent implementation or equivalent
  higher-precision path. Without one, report observed differences, repeatability
  or invariant checks and leave other accuracy claims unverified.
</reference>
```

## Diagnose a difference

Use the [diagnosis method](../../precision-repair/SKILL.md#diagnose).

```xml
<entry>
  An in-scope comparison trigger or reproducible failure on a supported native
  path has its inputs and evidence recorded.
</entry>

<action>
  Reuse the trigger and run the smallest check that separates remaining causes.
  Treat localization as progress unless it meets the agreed scope.
</action>

<output>
  Update issues.md with the classification, checked boundary, ruled-out
  explanations and evidence. Hand off the reproducer and focused check for repair.
</output>

<next>
  Correct setup or capture and repeat affected checks. Explain expected numerical
  differences. Route supported defects to authorized repair. For unresolved causes,
  continue while a check or pending result can advance them; otherwise deliver the
  blocker and next check. Keep independent issues moving before final delivery.
</next>
```

## Repair and verify

Use the [repair method](../../precision-repair/SKILL.md#patch-and-verify).

```xml
<entry>
  Repair is authorized and the defect has evidence, a reproducible trigger
  and a baseline (the original failure for an execution defect).
</entry>

<action>
  Test one candidate at a time. Default to the original small native case;
  extend only for missing coverage or an agreed delivery requirement.
</action>

<output>
  Record each candidate and its evidence in issues.md. Mark a fix verified
  when the focused regression and original-case check both pass.
</output>

<next>
  After fixing a preflight blocker, re-enter comparison for any outstanding
  numerical checks. Otherwise deliver the verified scope. On failure, return
  to diagnosis or try another supported candidate within budget.
</next>
```

## Local delivery

```xml
<entry>
  No required check remains actionable within authorization and budget.
  Follow active in-scope runs and finish available checks before final delivery.
</entry>

<action>
  Reconcile native execution, comparison, diagnosis and repair verification
  separately against the confirmed scope and actual evidence.
</action>

<output>
  Mark each stage complete, partial, blocked or not applicable. Use not
  applicable only when the stage was outside scope or its trigger was not
  met after the prerequisite check completed. For example, a CPU slice does
  not complete an agreed dual-native comparison. Give tested configuration,
  coverage, result, runnable setup and evidence links. For open issues, link
  issues.md with the blocker, trigger, next check, resources and remaining
  original-workload verification. Leave acceptance unset without criteria
  or assessment. Provide reports, bundles or offline replay when requested.
</output>

<next>
  Output “NumericBridge 链路已完成” only when all required, applicable stages
  have completion evidence. Then, or after a shareable partial delivery, offer
  community contribution independently of the local result.
</next>
```

## Community contribution

Use the [community record guide](../../precision-compare/references/community-records.md).

```xml
<contribute when="local delivery has a shareable result or a contribution is being resumed">
  Honor a prior decline or no-share instruction. Prepare an anonymized draft with
  minimum evidence and a PR draft for any verified patch intended for upstream.
  If prepare_community_submission returns invalid, correct its findings or
  report what blocks the draft. Only on ready, show the exact preview and
  destination and call submit_community with its draft ID and digest. Its form is the only approval: share the anonymized result so
  others can reuse it. Decline or dismissal ends the offer.
  If submission is unavailable, retain the local draft and report its path and
  blocker. A receipt means pending review; call the result searchable only when
  the deployed index returns its IDs. Maintainers handle publication and upstream
  follow-up. Contribution status does not change local delivery status.
</contribute>
```

If MCP is unavailable, keep an anonymized draft and use the
[local preview command](../../precision-compare/references/community-records.md#local-preview).
