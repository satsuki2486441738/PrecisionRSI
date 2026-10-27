# PrecisionRSI runner

A bounded task executor and an independent improvement stage over the existing workflow tools and an external agent harness. Run from the checkout with Python 3.10+; the runner uses the standard library. Workloads and harnesses supply their own dependencies.

```sh
python -m precision_rsi run task.json --workspace projects/my-task/session
python -m precision_rsi iterate series.json --workspace projects/my-stream/session
python -m precision_rsi improve improve.json --workspace projects/my-improvement/session
python -m precision_rsi evolve improve.json --workspace projects/my-evolution/session
python -m precision_rsi activate VERSION_ID --store projects/memory
```

## Task execution

Task paths are relative to the task JSON. `plan` uses the existing [workflow interface](../skills/precision-compare/scripts/workflow.py) for native preparation, reproduction, collection and task-owned checks. The agent edits implementations; the host executes and verifies the original workload.

```json
{
  "schema_version": 1,
  "family": "fixed-history-inference",
  "goal": "Reproduce the workload, compare its results and repair confirmed defects.",
  "plan": "native-plan.json",
  "budget": {"seconds": 1800, "agent_calls": 4},
  "agent": {
    "argv": ["dsh", "--profile", "headless", "--json", "{prompt}"],
    "timeout": 600,
    "usage_format": "dsh-jsonl"
  },
  "protected": ["verify.py", "inputs.json"],
  "memory": {"directory": "../memory", "tags": ["torch", "inference"], "limit_chars": 4000}
}
```

`family` groups related tasks for data separation. `{prompt_file}` supports adapters accepting a UTF-8 file. `agent.execution` selects a [record_run execution profile](../skills/precision-compare/scripts/record_run.py). Run the scheduler where the harness and workload paths are available; native remote stages retain their execution adapters.

`loop.json` retains time, calls, version, native receipts and raw harness output. Time/call budgets persist across invocations. Saved patches return to verification even after harness timeout or failure, within the remaining budget. Unchanged completion calls no model; unchanged failure after an action stops for lack of progress. The current system version is pinned for each new task; replay preserves that version. `status` inspects without executing.

## Improvement

```text
Task evidence → Select limitation → Build candidate → Independent evaluation → Activate version → Subsequent tasks
```

The separate improvement agent receives bounded summaries of completed and incomplete tasks, including tool counts, repeated commands and file mentions, with paths to detailed evidence. Its objective is task completion and lower cumulative time/token use, including construction, while preserving correctness and coverage. It chooses the bottleneck, method and update kind, then builds a minimal update or returns no update. The host prepares datasets, evaluates runtime bundles and activates accepted versions. Each round has its own budget inside the series budget.

```json
{
  "schema_version": 1,
  "tasks": ["task-a/task.json", "task-b/task.json"],
  "budget_seconds": 10800,
  "improvement": "improve.json"
}
```

An improvement configuration:

```json
{
  "schema_version": 1,
  "store": "memory",
  "budget_seconds": 600,
  "agent": {"argv": ["dsh", "--profile", "headless", "--json", "{prompt}"], "timeout": 480, "usage_format": "dsh-jsonl"},
  "trigger": {"min_new_tasks": 2, "on_failure": true},
  "max_updates": 4,
  "summary_limit": 12,
  "held_out_families": ["independent-inference"],
  "evaluation_plan": "evaluate-template.json"
}
```

A series supplies prior task states automatically. Standalone `improve` declares `evidence: ["/work/task/session/loop.json"]`; `{state: path, family: name}` can declare families for older traces. Paths resolve relative to the improvement configuration. New evidence triggers a round after the configured task count or an incomplete task. Replays reuse agent/evaluation receipts. Without `evaluation_plan`, runtime updates remain candidates.

`evolve` repeats the independent improvement stage with aggregate evaluation feedback. Its configuration adds `round_seconds` (default 900), `max_rounds` (8) and `patience` (3); `budget_seconds` covers all construction and evaluation rounds. Each round records candidates, measurements, raw usage and the next round's feedback, including construction costs and the declared reuse horizon. Failed bundle preparation retains candidate interfaces, artifacts and usage for revision. Replays reuse finished rounds; interrupted calls consume their reserved budget. No update, a repeated candidate, consecutive rounds without measured progress, or the budget ends the run. Evaluation templates may use `{round_workspace}` for isolated candidate workloads. Repeated evaluation tasks provide development feedback; a final generalization claim needs fresh tasks.

The agent writes `{"updates": [...]}` at the supplied output path. Every update has `kind`, `content` (short rule/interface), `tags`, `artifacts` (absolute persistent paths), and optional `benefit`/`replaces` (base member IDs). The host records provenance and base version.

| Kind | Artifact and use |
|---|---|
| `tool` | Reuse existing modules first; extract missing common logic into a callable module/CLI and regression. Consumers adapt inputs and invoke it. |
| `context` | A concise applicable rule; artifacts optional; retrieve within the context budget |
| `procedure` | `entrypoint` names a declared JSON artifact with `instruction` and/or `skill_paths`; referenced skill files are declared artifacts |
| `data` | JSONL examples referencing indexed task IDs; validated exports become `ready` datasets |

Procedure overlays change agent instructions and skill routing. Native task plans and correctness criteria remain task-owned. Scheduler implementation changes and agent-weight training are future extensions.

## Data and evidence

[evidence.collect](evidence.py) indexes task results, declared families, stage outcomes, time, usage and artifact identities. Full logs remain outside summaries. [dataset.build](dataset.py) accepts JSONL records:

```json
{"source":"INDEXED_TASK_ID","messages":[{"role":"user","content":"Task input"},{"role":"assistant","content":"Observed response"}]}
```

The exporter verifies source artifacts, assigns the source task outcome, deduplicates conversations and keeps each family in one split. Defaults are deterministic 80/10/10 family hashing; `splits: {family: "train"}` can override it. External held-out families are excluded. Labels describe whole-task completion, not correctness of every intermediate message; incomplete traces remain labeled incomplete. Data readiness establishes provenance and partition checks. Runtime benefit requires a later consumer/training evaluation.

## Versions and evaluation

Versions bind tools, memory, procedure overlays, artifact hashes, evidence families and runner/skill source identities. `replaces` removes superseded base members. Evaluation compares that exact bundle with its parent on disjoint task families. Accepted versions become current for subsequent tasks. `activate` restores a previously admitted, unchanged version; `activate empty` clears active memory.

An evaluator template uses `{version_id}`, `{baseline_version}`, `{store}` and `{construction_file}`. Its workflow declares `workload: {purpose: "held_out_evaluation", version_id: "{version_id}", baseline_version: "{baseline_version}"}`. The final output named `evaluation` binds these IDs and contains `mode`, `scope`, `families`, `baseline` and `candidate` measurements. Existing task-owned evaluators can use [versions.evaluate](versions.py).

The supplied [benchmark runner](benchmark.py) executes both arms and measures them:

```sh
python -m precision_rsi benchmark suite.json --version VERSION_ID --baseline BASE_ID \
  --workspace projects/evaluation/session --time-budget-seconds 1800 \
  --construction-file /work/improvement/construction.json --output /work/evaluation.json
```

Suite JSON: `schema_version: 1`, `store`, and `cases: [{id, family, baseline: task_JSON, candidate: task_JSON}]`. Arms use independent workload copies with matching initial conditions (including missing implementations), agent and memory-selection settings. The candidate must change the retrieved interfaces for that case before either arm runs. Run its command from the checkout in an evaluator workflow; `--output` supplies the named `evaluation` artifact. A case may replace `baseline` with `baseline_receipt: "/work/previous/0/baseline/result.json"`; reuse checks family, version, runtime, initial conditions and retained evidence. Each measured arm is retained for replay.

Measurements include native completion, time and raw usage. Construction time is included once by default; `amortize_over_tasks` declares an explicit reuse horizon. Known token totals include allocated construction usage. Optional `pricing` uses `input_per_million`, `cached_input_per_million`, `output_per_million`; configure improvement and task pricing separately. Unknown cache attribution/cost stays unknown. Closed admission preserves correctness and avoids measured time/token/cost regressions with at least one strict gain, or improves correctness. Evaluators own task selection, coverage and measurement validity.

The original optional single `proposal.json` from completed tasks and candidate-level `evaluate` CLI remain available. Versioned improvement evaluates the runtime bundle together. Operator search and agent-weight training are not implemented.
