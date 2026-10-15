# Architecture

## Skill package

The portable package is `skills/precision-skill/` in the source repository. Its directory name matches the frontmatter `name`. Copy that complete directory into a client's skill location under the same name.

```text
precision-skill/
  SKILL.md
  references/
    architecture.md
    workflow.md
    environment.md
    inference.md
    training.md
    comparison.md
  scripts/
    compare_scores.py
```

This follows the [Agent Skills specification](https://agentskills.io/specification): metadata supports discovery, `SKILL.md` provides the working instructions and direct reference links, and focused references are loaded for the task at hand. `references/` is the specification's convention for supporting documentation. The package uses no client-specific tool requirement. Repository README files and maintainer tests live outside the installable package.

## Responsibilities

| Layer | Responsibility | Handoff |
|---|---|---|
| Agent workflow | Establish conditions and scope, review environments/provenance, organize tasks, investigate gaps, and report. | Bounded tasks and evidence references described in [workflow](workflow.md). |
| Deterministic validation | Check supported identities, semantics, state, coordinates, coverage, and metrics. | Frozen contract and side artifacts; offline structured report. |
| Native collectors | Prepare and replay state/inputs; execute native library APIs; save effective conditions and observations. | Inputs, observations, environment, rank status, and field evidence. |

Compare native end-to-end stacks under common conditions. Internal operation-by-operation equality is a separate diagnostic question. A client loads instructions and presents results; collectors and the core remain independent of its model API or proprietary tools.

## Reusable capabilities

| Capability | Inputs and observations | Alignment |
|---|---|---|
| Forward/scoring | Fixed inputs and model state; scores or tensors | Tokens, positions, masks, distribution, dtype. |
| Cache/stateful execution | Initial state and fixed history; outputs/state transitions | Coordinates, layout, lifecycle, native/injected scope. |
| Objective/backward | Common dependencies; objective and gradients | Reduction, normalization, accumulation, scaling, communication, clipping. |
| Optimizer update | Parameters and optimizer initial state; before/after values | Formula, hyperparameters, step, master weights, state mapping. |
| Sampling/rollout | Initial state and sampling conditions or frozen trajectories | Actual randomness, termination, divergence, injection scope. |

Inference uses scoring/state capabilities, SFT adds objective/backward/update, and RL adds objective-specific trajectory dependencies. Distinguish planned, available, and actually executed capabilities. The bundled core implements the full-forward scoring subset.

## Workspace ownership

Keep private executions in a workspace independently of the installed package. For existing deployments, the workspace may remain alongside the skill files; preserve actual collector paths when migrating documentation.

```text
projects/<project>/
  scripts/                         # When multiple cases share implementation
  tests/
  environments/<side>/<version>/  # Declarations, locks, setting.yaml
  cases/<case>/
    contract.json
    <side manifests>
    README.md
    WORK_LOG.md
    scripts/
    tests/
    runs/<run-id>/
```

Framework code and maintainer tests belong in the source repository. Case-specific collectors/tests stay with the case; move them to project or framework scope only when that reuse exists. Environment declarations can serve multiple cases. Model weights, source datasets, and installed environments stay at their resource locations.

Each case has one current guide and one work log. Runs hold observations and logs, not copied guides. Use `<side>-<task>-<UTC YYYYmmddTHHMMSSZ>` for new run directories, with side `gpu`, `npu`, or `compare`; create exclusively and preserve completed artifacts. Save a new validation/comparison in a new run. A timestamp used to migrate an old artifact is labeled `migrated` and is not its execution time.

## Data boundaries

| Record | Contents |
|---|---|
| Request | Task, stacks, hardware, available environments/artifacts, question, and scope. |
| Shared contract | Semantics, input/state identities, numerical policy, mappings, reference, metrics/thresholds, required coverage. |
| Capability/scope record | Planned/available/executed capabilities, permitted actions, devices/budgets, and stopping conditions. |
| Provenance and input inventory | Sources, revisions, selected data, token/mask/position arrays, state/randomness, and mappings. |
| Side evidence | Actual observations and coordinates, effective conditions, environment association, counts, status, and injection scope. |
| Report | Execution, comparability, metrics/status, coverage, blockers, evidence, hypotheses, and implementation versions. |

Shared target releases or approved version pairs belong to case conditions. Actual releases/builds and platform configuration belong to side environments. Follow [environment](environment.md) for declaration-to-run association and resource reuse. Refer to declarations via `side_config.environment_config` and to actual observations via `environment`; resolve each path against its containing JSON.

The [comparison reference](comparison.md) owns schema-v2 handoff fields, metric formulas, evidence rules, and report states. The Python module docstring is the authoritative implemented field contract. Keep unsupported facts and their sources available for a case-specific adaptation rather than changing their meaning to fit the score schema.
