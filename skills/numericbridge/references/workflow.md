# NumericBridge workflow

## Stages and handoffs

```xml
<handoff>
  Confirm, retrieve, prepare and compare. Diagnose and verify when in scope,
  then optimize if enabled. Before launching, put known dependent commands in
  one workflow plan: resource readiness, native preflight, capture and comparison.
  Declare stage outputs and coverage checks from the task recipe. Keep bounded
  waits in the launcher; confirm the device binding again at native launch.
  Run the available chain in the same turn. A preflight or progress update does
  not pause it. If one side is blocked, advance independent work and fallbacks.
  Keep the confirmed request, plan, workload and runtime identities, evidence,
  remaining budget, issue record and exact next action for resume.
</handoff>
```

```xml
<issue_record>
  Create issues.md in the local case workspace when a finding changes the
  execution path or affects the result. Record it when found, including setup
  blockers, then update the same entry after diagnosis and each attempted change.
  Keep one current conclusion per issue: its status, affected configuration,
  verified scope, candidate outcome and open checks. Update it at each handoff;
  append attempts and link receipts, patches and measurements below it. Keep
  failed attempts and distinguish a workaround from repair of the original path.
</issue_record>
```

## Request and execution inputs

Start from the [confirmed request](request.md) and authorization.

```xml
<authorization>
  Carry authorization across stages and skills. Resolve routine setup, retries
  and evidence collection within the agreed scope and limits. Ask only for
  required inputs the agent cannot resolve or actions that require new approval.
  Before launching each native side, resolve placement, host or queue, launch
  method, resource limits and budget, checkpoint and revision, runtime and settings.
</authorization>
```

## Route community results

If the community MCP cannot be reached or cannot read its index, run the bundled
[search CLI](../../precision-compare/scripts/community_search.py) with the same
fields, using `--hardware`, `--model`, `--library`, `--task` and optional
`--version` or `--model-size`.

```xml
<retrieve>
  Call search_community with hardware, model, library (the tested framework)
  and task (inference, sft or rl). Pass version when fixed and model_size when
  needed. Resolve missing or invalid search fields before retrying the request.
</retrieve>
```

```xml
<retrieval_handoff>
  Record the returned status and index identity. If both paths are unavailable,
  record the lookup as unavailable. Continue to environment preparation after
  a successful lookup, including no-records, or when the lookup is unavailable.
  A case match supplies context, even when recommendation is null. Check the
  recorded versions, triggering settings and verification scope. For an
  applicable solution, reuse a matching local receipt or run the smallest
  target check that can confirm it. Otherwise continue from the confirmed
  request. A trial candidate can inform an enabled optimization; do not select
  it as the default. Retain user-fixed versions.
</retrieval_handoff>
```

## Prepare and compare

### Prepare the environment

Use [framework-adapt](../../framework-adapt/SKILL.md#execute) when deployment or
backend adaptation is needed. See its [native preflight contract](../../framework-adapt/SKILL.md#native-preflight)
for the minimum workload and required evidence.

```xml
<prepare>
  Resolve the measured stack from the confirmed request and retrieval result.
  Keep the target configuration separate from the reference configuration. For
  each side needing a native run, resolve placement, launch method, resource
  limits, source and runtime identity, model and data state, and the settings
  that affect the requested path. Reuse a runnable configuration and matching
  preflight receipt when their identities and scope still match.
  Use framework-adapt for deployment or backend adaptation. In an existing
  environment, verify device binding and run preflight if no matching receipt exists.
</prepare>
```

```xml
<prepare_handoff>
  Before capture, return the target and reference labels, effective execution
  JSON or its artifact path, source/runtime/resource identities, workload and
  preflight receipt, and any uncovered or blocked paths. A preflight pass means
  the planned native path is ready for capture. If target preflight fails, apply
  the recovery rule below before retrying. If still blocked, report the cause and
  leave that target path unverified. If only the native reference is blocked,
  apply the reference fallback below and record its coverage.
</prepare_handoff>
```

```xml
<preflight_recovery when="preflight reports failure or its final summary is missing">
  Use next_action.run and declared_outputs to locate landed evidence. Check the
  exact native owner's state before relaunching. If still active, follow that
  owner. If the native workload succeeded and only the wrapper failed, preserve
  the failed receipt. In a new recovery step, verify native terminal success,
  the planned input/runtime/device binding, and output identities and coverage
  against the original checks. Resume from the new verified receipt without
  replaying that native workload. If evidence is missing, recover or rerun only
  the missing checks after confirming the old owner is terminal. A landed
  file or failed wrapper is not a pass receipt.
</preflight_recovery>
```

### Run the comparison

Use [precision-compare](../../precision-compare/SKILL.md#collect) to choose the
task recipe and comparison tools. Use the [fixed 100-example batch](../../precision-compare/examples/precision-batch-v1.json)
when no dataset is specified.
The repository path is `skills/precision-compare/examples/precision-batch-v1.json`.
See the [dataset guide](../../precision-compare/examples/minimal-precision-batch.md) for task views.

```xml
<compare>
  Align inputs, initial state and coordinates. Declare intended differences.
  If no dataset is specified, load examples/precision-batch-v1.json from the
  precision-compare Skill root. Use all 100 events in the same order on both
  sides for formal comparison. For RL online, keep event IDs, prompts and order
  aligned while allowing each side to produce its native samples; pair only
  matching histories. Task recipes define grouping, sampling, update counts and
  when to use fixed RL replay.
  Cover the requested stages; expand only to exercise an uncovered path, isolate
  a difference or meet the delivery scope. Use triggering examples for
  diagnostic replay and record any injected or recomputed dependencies.
</compare>
```

```xml
<comparison_handoff>
  Return the workload and batch identities, selected stages, target/reference
  labels, expected and paired observations, missing or unavailable observations,
  metrics and their coordinates, execution receipts, and evidence paths. Keep
  native and offline-derived dtypes distinct. A successful comparison command
  means metrics were computed; it does not mean the user's tolerance was met.
  Leave acceptance unset when the user supplied no criteria.
</comparison_handoff>
```

### Choose a baseline and reference

```xml
<reference>
  For cross-platform diagnosis, prefer executable access to both platforms.
  Reuse matching captures and return to either side only for missing observations.
  With only the target accelerator, run the native case there and independently
  recompute planned checkpoints on CPU, using FP64 where suitable. Record exactly
  what the reference covers. Two configurations reveal differences; judge
  accuracy using an analytic result, a validated independent implementation or
  an equivalent higher-precision path. Without a suitable reference, report any
  repeatability or invariant checks and leave remaining accuracy unverified.
</reference>
```

## Diagnose a difference

Use the [diagnosis method](../../precision-repair/SKILL.md#diagnose) for an
in-scope difference that warrants investigation.

```xml
<diagnose when="an in-scope difference needs investigation">
  Apply the diagnosis handoff's stop conditions. Keep diagnosis open while it
  calls for a next check, even if all comparison checks are complete. Move to
  repair for a supported defect within scope. If diagnosis cannot proceed,
  deliver it as partial. For comparison-only work, skip diagnosis and deliver.
</diagnose>
```

## Repair and verify

Use the [repair and verification method](../../precision-repair/SKILL.md#patch-and-verify).

```xml
<verify when="a supported candidate fits the agreed scope">
  If verification fails, return to diagnosis or try another supported candidate
  within budget. If neither can proceed, deliver the evidence and unresolved checks.
</verify>
```

## Optimize precision

```xml
<optimize when="optimize_precision is enabled and a matching baseline is measured">
  After any required repair is verified, test the candidate against the current
  baseline on the same small case and reference. Recheck affected paths, agreed
  metrics and costs; inspect regressions across relevant samples, not just the
  trigger. Stop at a verified improvement within the limits, or when candidates
  or budget run out. If none qualifies, deliver the starting configuration.
</optimize>
```

## Resume and deliver

Use [execution receipts and plan checks](../../precision-compare/references/comparison.md#shared-tasks).
For repeatable command chains, declare required evidence as output `checks`
in the existing workflow plan. Follow `next_action` within scope; `completed`
covers only the checks declared in that plan.

```xml
<deliver>
  Before final delivery, reconcile every required check with its evidence.
  Report native execution, aligned precision comparison, and any diagnosis
  and repair verification separately. If a required check is still open,
  continue if an authorized action or wait can advance it within budget.
  Otherwise deliver a partial result with the missing evidence, concrete blocker
  and next action. A completed evaluation does not close an unresolved issue or
  a candidate with only partial improvement. Keep it open in issues.md and link
  it from the delivery. Record the trigger and reuse paths, the next focused
  check (command or adapter and inputs when known), its required resources, and
  the original-workload checks needed to close the issue. In a batch, prefer an
  actionable open issue with a reproducible trigger over starting another case
  when the remaining authorization, resources and budget permit.
  Report acceptance separately; leave it unset without criteria
  or an assessment. Include each candidate's tested configuration, runnable
  setup and issue record path when available. On follow-up, rerun only affected
  scope or explicitly requested fresh checks. Produce extra reports, bundles
  or offline replay on request.
</deliver>
```

Load the [community record guide](../../precision-compare/references/community-records.md)
when preparing a shareable result.

```xml
<contribute when="delivering shareable results or resuming a contribution">
  Honor a prior decline or no-share instruction. Prepare a local preview of the
  anonymized results and minimum evidence with prepare_community_submission.
  Show the preview and destination with the result, then call submit_community
  with its draft ID and digest. Its form is the sharing offer and the only
  confirmation: share the anonymized result so others can reuse it. A refusal
  or dismissal ends the offer; finish local delivery either way.
  If submission is unavailable, retain the local draft, give its path and
  report the blocker. On receipt, report the community receipt ID and that it
  is awaiting review. Community maintainers handle publication and upstream follow-up.
  Report "searchable" only when the deployed index returns the new record IDs.
</contribute>
```

If the contribution MCP is unavailable, keep the anonymized input JSON as a
draft. From the repository, validate it with
`python -B -m community.records.submission --input submission.json --output preview.json`.
