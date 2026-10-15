# Diagnostic workflow

## From request to report

1. **Discover facts and resolve decisions.** Use the request, configurations, and artifacts to determine the scenario, stacks, resources, and existing approvals. Ask together for decisions needed at the current stage, with a recommendation and its consequences.
2. **Freeze conditions and collection requirements.** Agree on semantics, required coverage, metrics, and thresholds or their absence. Specify saved inputs, coordinates, observations, effective conditions, completion records, and who collects them.
3. **Prepare environments.** Apply the shared-release and execution-association rules in [environment](environment.md). Reuse suitable environments and resources. Resolve a version exception before collecting or comparing with it.
4. **Run and accept a minimal case.** Exercise the native execution and collection path together. Check replay, saved observations, references, required evidence, and rank completion before increasing scale. Reuse matching existing results.
5. **Compare end to end.** Check identity, state, pairing, finite values, and coverage before computing metrics. Resolve comparability issues before investigating model internals. Without thresholds, report differences as diagnostics.
6. **Localize an observed issue.** Start with an anomalous case, position, or stage. Add the smallest useful observation, narrow to a module/operator when warranted, and regress the original end-to-end case after a change.
7. **Report and stop.** Separate execution, comparability, numerical status, and coverage. Preserve every blocker and artifact location. End when the question is answered, further investigation lacks evidence, or the execution budget is reached.

On resumption or migration, recheck the previous blocker with a bounded probe. Distinguish device visibility, socket permissions, DNS, and site connectivity. Reuse valid resource and environment records instead of repeating preparation.

Either GPU or NPU can establish a single-side baseline first. Execution order does not choose the reference direction. The two sides can run independently and exchange artifacts offline. Reused historical results undergo the same version and frozen-condition review as new results.

## Authorization, failures, and budgets

- Within the authorized task, resolve engineering issues such as paths, version-specific APIs, serialization, and installation mechanics. Preserve the agreed computation semantics.
- Obtain a decision before changing specified versions, model, inputs, dtype, scoring semantics, backend, devices, or budget. Installation, external connections, and data transfers follow the user's authorized scope. Reuse earlier approvals that cover the action.
- Stop a failed attempt and preserve its outputs. An evidence-based engineering correction may be retried within the approved budget. If one correction leaves the same blocker, report it. Data or semantic failures require fixing the cause rather than dropping samples or changing precision.
- Specify devices, scale, output location, cumulative device wall time, and stopping conditions before a new execution task. Include loading, failed attempts, and device diagnostics. Track installation/download budgets separately. Check the remaining budget before launch or retry.
- Record the external timeout, communication timeout, and relevant backend timeouts. Set the external limit from the remaining cumulative allowance. Interpret warnings using exit status, rank completion, and observation validity.
- Propose optimizations with evidence, modification scope, and acceptance criteria; obtain approval and regress the original case. Performance claims need an appropriate performance comparison.

A nonzero difference alone does not justify a full dump. A user-directed investigation can begin from a stated hypothesis, with explicit observations and scope. Keep native behavior and controlled diagnostic variants distinguishable. Root-cause claims require causal evidence beyond a local mismatch.

## Task handoffs

Simple work can combine inventory and validation. Split larger work when useful, supplying the relevant files/fields, frozen conditions, permitted actions and edits, devices/budget, expected output, and stopping conditions.

| Task | Deliverable | Next step |
|---|---|---|
| Inventory | Artifact locations, references, versions, scope, existing acceptance, and missing items | Validate the relevant artifacts. |
| Validation | Deterministic checks, evidence locations, results, and core report | Investigate required gaps. |
| Gap investigation | Facts found, classification, evidence limits, and minimal remedy | Resolve semantics with the user or adapt the tool as needed. |
| Collection | Inputs, observations, environment/status, valid references, and acceptance | Expand only after acceptance. |
| Comparison | Metrics, coverage, and all blockers | Diagnose an anomaly or finish. |
| Localization | Hypothesis, additional evidence, local reproduction, and regression | Stop without new evidence or at the budget limit. |

Delegated tasks return scope, actual actions, results, evidence locations, gaps, and next steps. The coordinating agent maintains the case work log. Read-only tasks return findings directly; tasks with write scope save artifacts within that scope.

## Collection and acceptance

Before execution, agree where each required fact will be saved and how it will be checked. Reuse the existing contract, `actual`, `evidence`, `environment`, `status`, and case records. Adapt a collector when its interface cannot express or replay the required semantics.

- Save actual replayable inputs, positions, masks, target mappings, and their selection/provenance.
- Confirm assigned devices are idle. Record process-visible IDs, physical IDs, board/chip mappings, and visibility variables. Use the ID namespace required by the runtime; verify a new platform mapping with a minimal device operation.
- Save observations that support pairing and recomputation. Summaries and internal dumps serve other purposes and do not replace those observations.
- Collect effective modes, precision stages, and the environment/completion status of every expected rank through the native model interface.
- Accept a small run by reading its artifacts, checking coordinates and references, and checking the current validator's required fields and types.
- Investigate missing evidence in relevant existing records first. A null field alone does not require a rerun. Propose only the collection needed for a real gap; save new evidence in a new run.

Use already authorized idle devices for relevant tensor and forward/backward smoke checks. Choose checks for the actual risk rather than requiring an additional full CPU preflight.

## Case work log

Maintain `projects/<project>/cases/<case>/WORK_LOG.md`. Keep the task and completion summaries to about ten lines each. Record meaningful steps in short entries, with long commands and logs referenced by path:

```text
Step: action
Reason: evidence or requirement
Execution: key configuration or script/command location
Result: outcome and finding
Evidence: artifact or log path
Next: follow-up or decision
```

Record tool-test results separately from model execution evidence. Use completion notifications and existing acceptance records; repeat checks when changed inputs, implementation, or unresolved failures justify them.
