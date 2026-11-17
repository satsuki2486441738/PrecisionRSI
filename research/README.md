# NumericBridge research runner

This is an optional research experiment, not part of the normal Skill installation.
Use the [skills](../skills/precision-compare/SKILL.md) directly for comparison and repair.

A bounded task executor and an independent improvement stage over the existing workflow tools and an external agent harness. Run from the checkout with Python 3.10+; the runner uses the standard library. Workloads and harnesses supply their own dependencies.

```sh
python -m research run task.json --workspace projects/my-task/session
python -m research iterate series.json --workspace projects/my-stream/session
python -m research improve improve.json --workspace projects/my-improvement/session
python -m research evolve improve.json --workspace projects/my-evolution/session
python -m research activate VERSION_ID --store projects/memory
```

## Task execution

Shared execution is imported from [tools](../tools/README.md); RSI does not import Compare scripts by modifying the process search path. Public [community records](../community/records/README.md) are independent of this optional runner.

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

### DeepSeek context compaction

DSH 0.2.1-rc.2 supplies automatic compaction, tool-result pruning and `/compact`.
Add the checked-in [overlay](dsh-compaction.patch.json) after the model overlay
for both task and improvement harness invocations:

```sh
dsh --profile headless --patch MODEL_PATCH --patch research/dsh-compaction.patch.json --json "TASK"
```

The policy reserves 16,384 headroom tokens plus the routed output allowance,
triggers at `floor(min(window * 0.8, window - output - 16384))`, retains the
latest 12.5% of `window - output` and caps summary generation at 8,192 tokens,
including reasoning. At a 1,048,576-token window and 262,144-token generation
allowance, the trigger is 770,048 and the retained tail is 98,304. The adapter
must declare the served capacity. Generation allowance includes reasoning and
answer tokens; it is distinct from the auxiliary summary cap.
DSH's default 65,536-token headroom leaves no proactive compaction budget on a
65,536-token route. The overlay fixes that without changing server capacity.

Pruning precedes summarization; one overflow recovery and one additional
compaction attempt are allowed. Native checkpoints preserve goals, files,
errors, constraints and pending work; full evidence stays on disk. Compaction
cannot shrink the fixed prefix or split an indivisible history unit. Keep large
artifacts in files. Compaction is an auxiliary model call; account for its usage
from session evidence separately when the headless step stream omits it.

Run the optional real-DSH regression with `DSH_BIN=/path/to/dsh python -B -m
unittest discover -s tests -p test_dsh_compaction.py -v`. It uses an isolated
session and a local mock provider; no inference service or credentials are needed.

Tasks with private native checks supply `interface: {...}`: visible inputs, identifier mappings, invocation and required output fields. Delegation sends this object instead of host plan/protected paths. Keep those files outside the harness view; the host retains them for execution and integrity checks. Interface changes invalidate baseline reuse.

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
  "consumers": [{"id": "compare", "tags": ["inference", "precision-compare"], "limit_chars": 2200,
                 "interface": "Declared capture JSON and tensors -> coverage and numerical differences"}],
  "evaluation_plan": "evaluate-template.json"
}
```

A series supplies prior task states automatically. Standalone `improve` declares `evidence: ["/work/task/session/loop.json"]`; `{state: path, family: name}` can declare families for older traces. Paths resolve relative to the improvement configuration. New evidence triggers a round after the configured task count or an incomplete task. Replays reuse agent/evaluation receipts. A proposal batch is evaluated as one bundle: invalid updates block evaluation, while valid candidates remain available for revision. Benchmark preflight requires every added interface to be selected by at least one evaluation task. Without `evaluation_plan`, runtime updates remain candidates.

`evolve` repeats the independent improvement stage with aggregate evaluation feedback. Its configuration adds `round_seconds` (default 900), `max_rounds` (8) and `patience` (3); `budget_seconds` covers all construction and evaluation rounds. Each round records candidates, measurements, raw usage and the next round's feedback, including construction costs and the declared reuse horizon. The result separates `construction_usage` and `evaluation_usage`; `usage` sums both. Benchmark receipts expose raw evaluation `usage` excluding construction and reused baselines; missing evaluation measurements remain unknown. Complete proposals saved before harness timeout/failure are recovered after input/runtime integrity checks; malformed proposals or missing artifacts are rejected. Recovery preserves the harness outcome and full construction cost, and still requires independent evaluation before activation. Failed preparation retains candidate interfaces and artifacts for revision. Replays reuse finished rounds; interrupted calls consume their reserved budget. No update, a repeated candidate, consecutive rounds without measured progress, or the budget ends the run. Evaluation templates may use `{round_workspace}` for isolated candidate workloads. Repeated evaluation tasks provide development feedback; a final generalization claim needs fresh tasks.

A measured baseline task failure is an outcome, not a prerequisite failure for improvement. Retain its receipt, pass aggregate failure feedback to the improver and evaluate fresh candidate workloads. Failed benchmark arms include the task reason and up to three failed native stages with bounded terminal errors; these facts reach the next round without source frames or full logs. Accept correctness recovery before comparing costs among successful runs; retain construction and failed-task costs. Source evidence and independent evaluation families remain disjoint.

Declare `consumers` with public input/output semantics, actual tags and JSON character budgets; omit private captures, verifier inputs and baseline implementations. The improver receives these declarations before construction. The shared runtime selector rejects uncovered bundle members before launching evaluation; the benchmark rechecks against actual task configurations. Without declarations, selection is checked by the benchmark only. Harness launch failures and native `turn_end` errors stop evolution with `needs_action`; replay preserves that result. Task verification failures remain improvement feedback.

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

### Durable research state

`evolve` retains a full redacted round ledger in `research/view/ledger.json`.
Only the latest three feedback records enter each prompt; older outcomes stay
queryable. The prompt declares the memory CLI, so the researcher need not read
its implementation:

```sh
python -m research.research /work/run/research/view history --start 0 --count 5
python -m research.research /work/run/research/view state
python -m research.research /work/run/research/view history --newest --outcome incomplete
python -m research.research /work/run/research/view history --candidate CANDIDATE_ID
python -m research.research /work/run/research/view notes --set notes.json
python -m research.research /work/run/research/view plan --set plan.json
```

`state` returns counts, the last accepted round and current notes/plan in one
bounded read. History filters accept `accepted`, `unaccepted` or `incomplete`;
`unaccepted` includes incomplete rounds. Candidate IDs are full lowercase SHA-256
IDs. Pagination offsets apply after filtering and ordering; original round numbers
stay unchanged, so note references remain valid. No query changes memory or
promotes a hypothesis. Use the attached state first, query older evidence when
needed, and revise notes/plan when decisions change rather than on every tool call.

The prompt includes valid minimal notes/plan schemas; `--set` replaces the whole
document. Empty `evidence_rounds` is valid before any observed round exists.

Notes accept `observations: [string]`, `uncertainties: [string]`, and
`hypotheses: [{claim, evidence_rounds: [1]}]`. Plans accept
`next_experiments: [{intervention, falsified_if, evidence_rounds: [1]}]`.
Both are optional, bounded to 4096 UTF-8 bytes and at most 16 referenced rounds,
and validate round references. Current judgments reattach to each new round;
the round freezes its input context so replay survives subsequent note edits.
These are researcher judgments; ledger outcomes remain controller observations.

The researcher can include changed `handoff: {notes: ..., plan: ...}` beside
`updates` in its normal proposal; the controller validates and saves both without
another model or memory call. Invalid judgments leave earlier documents intact
and do not invalidate a candidate. Replaying a cached result never overwrites
newer memory. Without submitted judgments, their absence remains explicit.

Independently, the controller saves `handoff.json` at completed model steps and
construction/evaluation/review boundaries, including timeout. This bounded state
contains observed usage, proposal presence, unverified artifact filenames and the
next workflow action. It joins `state`, round context and snapshots. A hard kill
retains the last saved checkpoint, possibly still marked running; it does not
prove artifact correctness or bypass unfinished-session reconciliation.

For native DSH conversation continuity, add `"persistence": {"kind": "dsh"}`
to the improvement `agent` object. Its argv must forward DSH flags, use
`usage_format: "dsh-jsonl"`, and end with a standalone `{prompt}` or
`{prompt_file}`. Keep the compaction overlay configured as above. The adapter
adds a JSONL persistence overlay and resumes the same `--session-id` across
rounds, with a stable `research/view/work/` cwd and absolute per-round output
paths. Generic harnesses retain turn receipts without native conversation resume.
Standalone `improve` can join an explicitly supplied
`research: {root, binding, turn}`; `evolve` supplies this automatically.

Private `research/private/` holds the identity binding, native session logs,
cursor and per-turn receipts. Completed turns replay stdout without calling the
model; changed inputs or missing native state require explicit recovery. After
confirming an unfinished process has stopped, an operator may abandon its turn:

```sh
python -m research.research /work/run/research reconcile round-001 --reason "Process stopped; partial turn abandoned"
```

This records interruption, never retries that turn or grants more budget.
The controller can then continue with the retained failure evidence. A process
crash blocks automatic continuation until reconciled. Keep private state, host
checks and snapshots outside the researcher filesystem view using the execution
sandbox; the directory layout and CLI alone do not enforce filesystem isolation.
The session binding pins configuration and runner/skill bytes. Declare external
model/runtime settings in the frozen configuration; external services are not
snapshotted. Raw native events remain private. Report auxiliary compaction usage
as unknown when the headless stream does not account for it.

### Research snapshots

`evolve` and standalone `improve` seal private `.snapshots/` journals automatically:
initialization (`checkpoint-000`), round/harness start, submitted candidates before
evaluation, and committed results, including rejection and failure. Parent IDs
link boundaries. Objects are SHA256-addressed and deduplicated; manifests retain
exact bytes, executable flags, candidate provenance, active selection and admission
receipts. Platform sources include runner code, skills, references, the harness
overlay and maintenance dependency declarations. Model weights and datasets remain
verified resource references requiring binding; snapshots do not back them up.

```sh
python -m research.snapshots --journal /work/run/.snapshots list
python -m research.snapshots --journal /work/run/.snapshots verify SNAPSHOT_ID
python -m research.snapshots --journal /work/run/.snapshots restore SNAPSHOT_ID /work/new-fork
python -m research.snapshots --journal /work/run/.snapshots restore SNAPSHOT_ID /work/operator-archive --view operator
python -m research.snapshots --journal /work/run/.snapshots rollback SNAPSHOT_ID --store /work/memory
```

Researcher exports contain only declared `changes/` and `session/` files, including
the ledger, notes and plan. Native sessions, cursor and turn receipts are bound
under `audit/research/`; context coverage records whether native JSONL was exported. Operator
exports additionally include `platform/` and `audit/`, including the original-to-local
artifact mapping in `audit/bindings.json`. Keep journals/operator exports outside
the task agent's filesystem view. Neither view executes code, resumes submissions
or restores process/service readiness. Start/submitted boundaries require explicit
`--evidence` export and are not recoverable checkpoints.

`rollback` uses existing version admission checks and requires the same live
platform and intact candidate artifacts. It changes the active bundle for future
tasks; it does not rewind task workspaces or ongoing jobs. When original files or
platform have changed, restore into a new directory, rebind paths/resources and
prepare/evaluate a new version. Original IDs and absolute paths in archived JSON
remain unchanged. This is a detached fork, not transparent execution replay.

Automatic session coverage is the delivered prompt, submitted proposals and private
controller feedback/state. Full native DSH messages, compaction coverage and session
cursors are not yet exported automatically; `audit/context-coverage.json` records
that limitation. `snapshots.Journal.capture` accepts explicit allowlisted files or
JSON values in the four layers for future harness-native checkpoint exporters.
Do not treat a round checkpoint as a complete model conversation checkpoint.
Historical runs are not retroactively changed. Local rollback storage is not an
independent backup; preserve resource bindings and separate durable copies.

Versions bind tools, memory, procedure overlays, artifact hashes, evidence families and runner/skill source identities. `replaces` removes superseded base members. Evaluation compares that exact bundle with its parent on disjoint task families. Accepted versions become current for subsequent tasks. `activate` restores a previously admitted, unchanged version; `activate empty` clears active memory.

An evaluator template uses `{version_id}`, `{baseline_version}`, `{store}` and `{construction_file}`. Pass `{evaluation_budget_seconds}` as an argv value to nested evaluators; it supplies the remaining round budget. Reserve time within that limit for native verification and receipt writing. Its workflow declares `workload: {purpose: "held_out_evaluation", version_id: "{version_id}", baseline_version: "{baseline_version}"}`. The final output named `evaluation` binds these IDs and contains `mode`, `scope`, `families`, `baseline` and `candidate` measurements. Existing task-owned evaluators can use [versions.evaluate](versions.py).

The supplied [benchmark runner](benchmark.py) executes both arms and measures them:

```sh
python -m research benchmark suite.json --version VERSION_ID --baseline BASE_ID \
  --workspace projects/evaluation/session --time-budget-seconds 1800 \
  --construction-file /work/improvement/construction.json --output /work/evaluation.json
```

Suite JSON: `schema_version: 1`, `store`, and `cases: [{id, family, baseline: task_JSON, candidate: task_JSON}]`. Arms use independent workload copies with matching initial conditions (including missing implementations), agent and memory-selection settings. The candidate must change the retrieved interfaces for that case before either arm runs. Run its command from the checkout in an evaluator workflow; `--output` supplies the named `evaluation` artifact. A case may replace `baseline` with `baseline_receipt: "/work/previous/0/baseline/result.json"`; reuse checks family, version, runtime, initial conditions and retained evidence. Each measured arm is retained for replay.

Update tags are applicability requirements: all must be present in the consumer task tags. Retrieved JSON interfaces, including artifact paths, share the task's `limit_chars`. If retrieval yields no runtime difference, `selection.json` records consumer tags/budget, selected IDs and rejection reasons (`unmatched_tags`, `interface_budget`, etc.); the evaluator error includes the same diagnostic. `memory.select(..., diagnostics=[])` exposes these reasons directly.

Measurements include native completion, time and raw usage. Construction time is included once by default; `amortize_over_tasks` declares an explicit reuse horizon. Known token totals include allocated construction usage. Optional `pricing` uses `input_per_million`, `cached_input_per_million`, `output_per_million`; configure improvement and task pricing separately. Unknown cache attribution/cost stays unknown. Closed admission preserves correctness and avoids measured time/token/cost regressions with at least one strict gain, or improves correctness. Evaluators own task selection, coverage and measurement validity.

Each benchmark arm separates `execution` (measured task seconds, raw usage/tokens and cost) from `including_construction` (amortized seconds/tokens/cost used for admission). Legacy scalar admission fields retain their meaning; improver feedback and durable numeric memory use the explicit groups. Old receipts without execution timing keep that value unknown. Construction receipts are bound by content; changes require a new workspace.

The original optional single `proposal.json` from completed tasks and candidate-level `evaluate` CLI remain available. Versioned improvement evaluates the runtime bundle together. Operator search and agent-weight training are not implemented.

## Source layout and frozen versions

The former RSI runner now lives in `research/`; commands use `python -m research`.
Shared tool implementations are in `tools/`, and maintainer commands in `scripts/`.
Research snapshots and runtime identities include the shared tools. An operator
restore includes them so research imports work without the original checkout.
Changed paths or source identities require new affected verification or version
evaluation; existing checkpoints, runs and historical activation receipts are
retained as recorded and are not silently admitted under the new layout.
