# Execution workflow

## Prepare and run

- Reuse agreed conditions, saved inputs, valid observations and prior approvals. Record resolved choices in the case; do not ask the user to reconfirm them. Ask only when an unresolved choice affects the requested semantics or authorized scope.
- Before a new device run, establish the command, device allocation, output location, remaining cumulative budget and timeout. Count loading, failed attempts and cleanup. Use the project's environment manager and scheduler; verify actual device mapping and idleness.
- Save replayable inputs/state, observations, effective numerical settings, run-associated environment evidence and completion status for every rank. Preserve native execution paths; configuration intent alone is not runtime evidence.
- Validate each side before pairing: inputs, identities, states, coordinates, counts, finite values and completion. Reuse preparation evidence; missing a particular file is not by itself a reason to rerun.

## Failures and investigation

Preserve failed attempts. Resolve ordinary engineering issues within the existing scope and budget; rerun only the affected validation. Do not silently change the model, data, numerical policy or required coverage to obtain agreement. Stop when the budget is exhausted or an unresolved blocker prevents useful progress.

Begin localization at a discrepant position, tensor or stage. Collect the smallest additional evidence that can distinguish the suspected causes; a nonzero error alone does not require a whole-graph dump. Keep controlled diagnostic variants separate from native-stack results.

## Handoff and report

A handoff contains the shared conditions, observations, environment/launch evidence, validator and a short next-action note. Either side can execute first; execution order does not change reference direction.

Keep one case guide and a brief work log. Runs hold immutable observations and execution logs; new attempts/comparisons use exclusive directories. Separate software-test success, successful device execution and cross-device numerical agreement.

Report only the outcome, principal metrics, measured coverage, unresolved blockers and evidence link. Include detailed environment inventories or formulas by reference rather than repeating them in each response.
