# NumericBridge workflow

## Stages and handoffs

```xml
<handoff>
  Confirm, retrieve, prepare and compare. Diagnose and verify when warranted and
  in scope, then optimize if enabled. A stage handoff is a checkpoint: record
  its result, give a progress update if useful, and execute the next in-scope
  step in the same turn. If one side is blocked, continue independent work on
  the other side and use applicable reference fallbacks within scope.
  Deliver when the request is satisfied or no authorized action or wait can
  advance it within budget. For incomplete work, report the blocker. For resume,
  retain the confirmed request, stage, workload, source/runtime identities,
  evidence paths, issue record path when present, and exact next action.
  Reuse matching evidence.
</handoff>
```

```xml
<issue_record>
  Create issues.md in the local case workspace when a finding changes the
  execution path or affects the result. Record it when found, including setup
  blockers, then update the same entry after diagnosis and each attempted change.
  Keep the symptom, triggering configuration and workload, cause or open
  question, and each candidate's change and outcome. Link to existing receipts,
  patches and measurements. Keep failed attempts and open checks; distinguish a
  verified workaround from repair of the original path.
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
  Carry only applicable case, issue and verification references. Treat a non-null
  recommendation as a candidate stack and verify it on the target workload.
  With a null recommendation, continue from the confirmed request and explain
  insufficient evidence if the user asked for a stack recommendation. Retain a
  user-fixed version unless the user agrees to change it.
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
  the planned native path is ready for capture. If target preflight fails, resolve
  the environment or adaptation within scope. If still blocked, report the cause
  and leave that target path unverified. If only the native reference is blocked,
  apply the reference fallback below and record its coverage.
</prepare_handoff>
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
  Move to repair for a supported defect within scope. If the cause is unresolved,
  continue diagnosis while a focused check fits the budget; otherwise deliver
  the evidence and next check. For comparison-only work, skip diagnosis and deliver.
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
<optimize when="optimize_precision is enabled and the starting configuration is verified">
  Compare against the verified configuration using the same small case and
  reference. Recheck affected paths and measure the agreed metric and costs.
  Stop at the first verified improvement within those limits, or when candidates
  or budget run out. If none qualifies, deliver the verified starting configuration.
</optimize>
```

## Resume and deliver

Use [execution receipts](../../precision-compare/references/comparison.md#shared-tasks)
when resuming work.

```xml
<deliver>
  Reuse any comparison handoff for each candidate. Include its actual configuration,
  runnable setup when available, unresolved checks and the local issue record
  path when present. Report whether agreed criteria were met; leave acceptance
  unset without criteria or an assessment.
  On follow-up, rerun only affected scope or explicitly requested fresh checks.
  Produce extra reports, bundles or offline replay on request.
</deliver>
```

Load the [community record guide](../../precision-compare/references/community-records.md)
only after the user opts in.

```xml
<contribute when="shareable results have been delivered">
  Ask once whether to share anonymized results so others can reuse the findings.
  On opt-in, remove identifying and private details. Call prepare_community_submission
  and show its preview. Call submit_community with that draft ID and digest for
  user confirmation. Report the submission outcome and PR link if available;
  keep the local draft if submission cannot proceed.
</contribute>
```
