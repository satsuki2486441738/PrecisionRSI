# PrecisionRSI runner

The task loop runs native commands, calls an existing agent harness for missing adaptation/repair, and verifies the original workload. The improvement loop retains candidate experience, evaluates it on separately selected tasks, and exposes admitted tools/context to subsequent tasks. Operator search and agent-weight training are extension points.

Run from the repository checkout; the runner uses Python 3.10+ and the standard library. Native workloads and agent harnesses supply their own dependencies.

```sh
python -m precision_rsi run task.json --workspace projects/my-task/session
python -m precision_rsi status task.json --workspace projects/my-task/session
python -m precision_rsi iterate series.json --workspace projects/my-stream/session
```

## Task interface

Paths are relative to the task JSON. `plan` uses the existing [workflow interface](../skills/precision-compare/scripts/workflow.py); it declares native environment, reproduction, collection and task-owned checks. The agent modifies implementations; the runner reuses valid stages and reruns affected verification. The plan and declared `protected` inputs retain their identities. A changed task/evaluator uses a new workspace. Budget increases can resume the same workspace.

```json
{
  "schema_version": 1,
  "goal": "Reproduce the supplied workload, diagnose and repair confirmed numerical defects.",
  "plan": "native-plan.json",
  "budget": {"seconds": 1800, "agent_calls": 4},
  "agent": {
    "argv": ["dsh", "--profile", "headless", "--json", "{prompt}"],
    "timeout": 600,
    "usage_format": "dsh-jsonl"
  },
  "protected": ["verify.py", "inputs.json"],
  "memory": {"directory": "../memory", "tags": ["torch", "sft"], "limit_chars": 4000}
}
```

Use `{prompt_file}` instead of `{prompt}` for a harness adapter accepting a UTF-8 prompt file. `agent.execution` optionally selects a [record_run execution profile](../skills/precision-compare/scripts/record_run.py). Run the scheduler on the host where the harness and workload paths are available. Remote native stages use their existing execution adapters.

`loop.json` retains cumulative time, agent-call count, native results and raw harness output paths. The time budget includes native execution and agent calls; agent timeouts terminate the owned process group. Saved patches return to native verification even after an agent timeout or nonzero exit, within the remaining budget. Completion requires native checks. Unchanged completion calls no model; unchanged failure after an agent action stops for lack of progress. Optional `dsh-jsonl` usage counts completed Harness step events; unknown cache usage remains `null`. Other harnesses retain raw output with normalized usage `null`.

## Experience and iteration

An agent may leave `proposal.json` in its task workspace:

```json
{"kind":"tool","content":"Use /work/tools/capture.py --help for this capture interface.","tags":["torch","sft"],"artifacts":["/work/tools/capture.py"]}
```

Completed tasks retain proposals as candidates. Kinds are `tool`, `context`, `data`, `weights`; only tools/context currently support admission. Artifact paths are absolute and content-identified. Active memory includes only matching tags and unchanged artifacts within the configured character budget. Candidates and full traces are outside the prompt.

```json
{
  "schema_version": 1,
  "tasks": ["task-a/task.json", "task-b/task.json"],
  "budget_seconds": 7200,
  "evaluation_plan": "evaluate-template.json"
}
```

`iterate` executes the finite task list. After a task proposes experience, it renders the evaluator template with `{candidate_id}` and `{candidate_file}`, runs it, and continues with the resulting memory. Without `evaluation_plan`, proposals remain candidates. A replay reuses native/evaluation receipts.

The evaluator is a workflow plan with `workload: {purpose: "held_out_evaluation", candidate_id: "{candidate_id}"}`. Its final named output `evaluation` contains:

```json
{
  "candidate_id": "{candidate_id}", "mode": "closed", "scope": ["independent-task-id"],
  "baseline": {"correct": true, "seconds": 100, "cost": 0.2},
  "candidate": {"correct": true, "seconds": 80, "cost": 0.15}
}
```

The evaluator owns independent task selection, correctness/coverage checks, timing and usage-based pricing; it runs or reuses verified baseline/candidate measurements. Building and maintaining an improvement belongs in its cost accounting. Closed admission requires improved correctness, or preserved correctness with no measured time/cost regression and a strict gain. Optional cost must be supplied for both sides. An open evaluation instead compares higher `quality` at equal positive `budget`; operator search itself remains unimplemented.

Run an evaluation separately with `python -m precision_rsi evaluate ID --store MEMORY --plan EVALUATION.json --workspace EVALUATION_WORKSPACE --time-budget-seconds 600`.
