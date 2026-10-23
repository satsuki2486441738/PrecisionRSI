# Compare artifacts

## Tools

Use the entrypoint's common tool declarations directly. Query missing details without importing dependencies:

```sh
python scripts/tool_help.py --list                  # module + purpose
python scripts/tool_help.py compare_trees           # short declaration + signature
python scripts/tool_help.py --full tree_compare     # detailed contract + interfaces
```

Module names return short declarations; API names add current signatures. Script `--help` also uses short declarations. Qualify ambiguous APIs as `module:API`; batch queries share declarations. Modules with `__all__` expose only those APIs. When building new manifests/plans or integrating capture, query selected tools with `--full` directly for their schemas and constraints.

| Operation | Interface | Input → output |
|---|---|---|
| Discover interfaces | [tool_help](../scripts/tool_help.py): `--list` or API names | Names → short declarations/signatures; `--full` → detailed contracts |
| Compare arrays | [array_compare](../scripts/array_compare.py): `compare_arrays` | Named arrays / NPZ → equality, coverage, optional metrics |
| Compare state | [tree_compare](../scripts/tree_compare.py): `compare_trees`, `compare_checkpoints` | Nested state / Torch files → structure, metadata, numeric differences |
| Group results | [group_metrics](../scripts/group_metrics.py): `reduce_groups` | Coordinates, groups, reducer → global/per-group results |
| Compare tensor artifacts | [compare_tensors](../scripts/compare_tensors.py): `compare`, `--mode metrics` / `nonfinite` | Contract + two manifests → metrics or anomaly counts/finite-subset metrics |
| Compare target scores | [compare_target_scores](../scripts/compare_target_scores.py) | Contract + score records → NLL and logprob errors |
| Compare distributions | [compare_distributions](../scripts/compare_distributions.py) | Contract + manifests → top-100 union/tail metrics |
| Query remote distributions | [query_distributions](../scripts/query_distributions.py): `inventory`, `query`, `compare` | Source inventories + union queries → metrics; arrays stay at source |
| Compare assistant-body scores | [compare_scores](../scripts/compare_scores.py) | Cache-off scoring artifacts → metrics and execution checks |
| Validate artifacts | [validate_artifacts](../scripts/validate_artifacts.py) | Contract + one manifest → validity/coverage |
| Export / compare tensor views | [tensor_artifacts](../scripts/tensor_artifacts.py): `export`, `compare-views` | Rank inventories / view pairs → logical manifests / comparisons |
| Capture inference | [inference_capture](../scripts/inference_capture.py): `FixedHistory`, `derive_logprobs`, `export_records` | Fixed tokens + native logits → validated histories/scores |
| Plan internal capture | [msprobe_capture](../scripts/msprobe_capture.py): `plan_modules` | Modules + sizes → bounded L0 plan ([integration](layerwise.md#integrate-and-verify)) |
| Probe runtime | [runtime_probe](../scripts/runtime_probe.py): `probe_runtime`, `runtime_identity` | Imports, checks, selected sources → runtime evidence/identity |
| Record execution | [record_run](../scripts/record_run.py) | Literal argv + timeout → run receipt, logs, failure summary |
| Find prepared resources | [resource_catalog](../scripts/resource_catalog.py): `register`, `find` | Resource identities + successful receipt → verified lookup |
| Resume workflow | [workflow](../scripts/workflow.py): `run`, `status` | Dependency plan + workspace → reuse/execution/continuation |
| Package evidence | [evidence_bundle](../scripts/evidence_bundle.py): `pack`, `verify`, `extract` | Selected files / ZIP → indexed portable evidence |
| Replay evidence | [replay_bundle](../scripts/replay_bundle.py) | Bundle + replay specification → reproduced reports/checks |

Tensor comparison defaults to `mode=metrics`; `mode=nonfinite` selects category counts and paired-finite metrics. `--chunk-size` controls streaming reads; `--max-locations` caps nonfinite coordinate examples.


Adapters establish semantic correspondence, input/state and dtype identity. Equal names/shapes alone do not establish correspondence. Native mappings avoid file conversion; other formats use task-owned loaders and leaf adapters. A leaf adapter may return `{dtype: native_dtype, values: exported_array}` to retain dtype identity during conversion.

Array/tree comparisons require finite, exactly equal numeric values for `exact_equal`; optional metrics use FP64 arithmetic. Same-shape arrays with different dtypes retain dtype findings and support error metrics; `exact_equal` still requires matching dtypes. Findings are bounded examples; counts retain all differences. Check an independent task inventory for omissions shared by both sides. Add traversal only for requested details absent from reports.

For grouped comparisons, derive groups from task coordinates/settings and pass `compare_arrays` as reducer. Use `lambda a, b: compare_arrays(a, b, metrics=True)` for error metrics. Groups may overlap; incomplete groups have no aggregate, while complete groups remain available.

The [gradient example](../examples/compare_gradients.py) directly compares native CPU FP32/BF16 gradients (Torch and NumPy required).

Supporting modules provide [artifact readers](../scripts/artifact_io.py), [execution-condition checks](../scripts/conditions.py) and [target-score arithmetic](../scripts/score_metrics.py). Reuse them inside tools and task adapters; their module docstrings own the interfaces. Task adapters own native masks, logical mappings, losses and objective reductions.

For manifest-based tools, run from the skill directory. The score, distribution and tensor comparators share `--contract`, `--side-a`, `--side-b` and `--output`; the validator uses `--manifest` for one side's input. Use fresh output paths. For a worked collection task, see the [fixed-history inference example](../examples/fixed-history-inference.md).

```sh
python scripts/compare_tensors.py --contract gradients.json --side-a a.json --side-b b.json --output report.json
```

When preparing manifests, use the selected interface. Distribution matrix rows belong inside each `scores` descriptor, e.g. `scores: {path: rows.npy, row: 3}`; multirow arrays require an explicit row. Put equal conditions in `common` and task-selected differences in `side_conditions`; each rank records its actual settings and imported versions. [Condition checks](../scripts/conditions.py) compare these against the contract-owned rank inventory. For task-selected different initial parameter representations, declare per-side `state`; common `state` retains exact before-value checks. Implement missing task-specific integration and reuse the shared arithmetic.

Distribution/tensor schemas share [artifact_io.py](../scripts/artifact_io.py). Inspect implementation when the selected interface leaves an integration question unresolved.

Contracts set `thresholds: null`. Outputs contain metrics, coordinate counts and unavailable comparisons. Exit codes: 0=comparison metrics computed successfully, 1=unavailable comparison, 2=CLI/output error. Exit 0 does not establish that a user's accuracy tolerance is met.

## Shared tasks

Start from available environment and launcher declarations; check current allocation and changed runtime conditions together with `runtime_probe`. Keep provider-specific submission and device binding in the environment adapter.

Run a native command through `record_run` to retain complete logs and receive a compact result with failure diagnostics:

```sh
python scripts/record_run.py --output-root runs --side SIDE --task capture --timeout 300 -- PYTHON COLLECTOR ARGS
```

Its CLI returns log paths and byte counts; failures include the final 2048 bytes per stream (`--tail-bytes` adjusts this). For scheduled work, the adapter waits for the job's terminal result within its budget and propagates its exit status; a successful submission alone does not establish workload completion. Batch independent probes and use completion waits of 10–30 seconds for noninteractive commands where supported.

Continue the existing workload workspace across components. Reuse verified evidence unless inputs, runtime identity, scope or requested freshness change. For repeated dependency-linked commands, use a workflow plan declaring relevant inputs, outputs, checks and dependencies. Adapters take input/output paths and workload settings as parameters; derive inventories, phase boundaries and report labels from settings and computed fields. Stage-specific settings belong in stage inputs; changing the shared workload identity invalidates every stage. Its [docstring](../scripts/workflow.py) defines the interface.

```sh
python scripts/workflow.py run --plan task.json --workspace work
python scripts/workflow.py status --plan task.json --workspace work
```

`status` reviews reuse without launching commands. `run` both reuses valid results and executes missing work, so a known command chain needs one invocation. Continue with `next_action` only when execution stops for adaptation or repair.

For result-review requests, return verified retained outputs and their source paths. For fresh measurements, use `--refresh` on the requested capture or comparison stage(s); descendants rerun while valid preparation and unrelated stages remain reusable. Changed inputs invalidate affected stages automatically. Reopen instructions/interfaces when they change or an integration question remains. Task-owned correctness checks can route to repair.

Use the resource catalog when prepared environments/adapters need discovery across workloads. Register reusable resources with successful native command receipts. A single task can use its existing paths directly.

The [native Torch example](../examples/workflow_demo.py) demonstrates preparation reuse, a controlled repair and original-workload verification; unchanged replay launches zero native commands.

## Portable evidence

When portable evidence or replay is requested, use `evidence_bundle.py` to pack selected artifacts. Its index records original/exported SHA256 and byte counts. Optional literal replacements apply to UTF-8 text while preserving numeric arrays; keep the replacement file outside the selected directory. The tool docstring defines file selection and size limits.

```sh
python scripts/evidence_bundle.py pack SELECTED_FILES NEW.zip
python scripts/evidence_bundle.py verify NEW.zip
python scripts/evidence_bundle.py extract NEW.zip NEW_DIRECTORY
python scripts/replay_bundle.py --bundle NEW.zip --output-root runs
```

For replay, include `REPLAY.json` with named cases selecting a supported validator/comparator, relative input paths and expected report fields. Tensor cases may set `mode=metrics|nonfinite`. Preserve side identities and order. Replay uses the installed skill CLIs and retained arrays; expected-value checks establish reproducibility of those metrics. The [replay tool docstring](../scripts/replay_bundle.py) owns case fields and supported operations.

## Coordinates

Pair target scores by sample/target position/token, distributions by sample/prediction position, and tensors by logical tensor/element. In causal inference, prediction position $t-1$ predicts target position $t$. For `stage: observations`, declare each tensor's logical sampling `coordinates` in the contract and both manifests; retain sample/token selection, boundary, invocation and execution stage as applicable. Native names map to this identity in the adapter. Reconstruct logical coordinates from shards, packed batches and fused projections; retain tied-weight mappings and reduction tags. Check expected coordinates, masks and ranks, and verify converted initial values. Exclude padding, deduplicate equal replicas, and retain differing replicas as explicit views. Distinguish absent gradients from present zeros.

## Target-token scores

For two executions' logprobs $\ell_{a,i},\ell_{b,i}$ over $N$ valid targets, report each side's NLL and their absolute difference:

$$
\mathrm{NLL}_{s} = -\frac{1}{N}\sum_{i=1}^{N} \ell_{s,i}
$$

For $e_i=|\ell_{a,i}-\ell_{b,i}|$, report MAE, P99 and maximum with its coordinate.

Use the global valid-target count. Native loss retains its implemented reduction. Percentiles use all paired errors, linearly interpolated at $a(N-1)$; P99 uses $a=0.99$.

## Tensors

For paired tensors $a$ and $b$, use stable FP64 reductions:

| Metric | Definition |
|---|---|
| L2 error | $\lVert a-b\rVert_2$ |
| RMSE | $\lVert a-b\rVert_2/\sqrt{N}$ |
| Cosine | $a^\top b/(\lVert a\rVert_2\lVert b\rVert_2)$ |
| Maximum absolute error | $\max_i \lvert a_i-b_i\rvert$ |

Global metrics concatenate coordinates. Retain counts and signal/error norms. Actual updates are $\theta_{after}-\theta_{before}$, promoting stored values to at least FP32 before subtraction.

Cosine is undefined when either norm is zero. Represent undefined metrics as null with a reason.

Use finite target scores/tensors; distribution logprobs permit `-inf` for zero probability. For tensor NaN or Inf, use `compare_tensors.py --mode nonfinite` with the same contract and manifests. Report anomaly counts and paired finite coverage separately from finite-subset error metrics; matching nonfinite locations do not establish correctness. Query `tool_help.py --full compare_tensors` for detailed capture encodings and diagnostic semantics. Retain the original arrays for repair and evidence replay.
