# Compare artifacts

## Tools

Query selected interfaces without importing their dependencies or reading implementations:

```sh
python scripts/tool_help.py tree_compare array_compare
```

For already corresponding observations, use [array_compare.py](../scripts/array_compare.py):

```python
from array_compare import compare_arrays
report = compare_arrays(side_a_arrays, side_b_arrays, expected=logical_names)
```

For task-defined groups of named arrays (for example, layers, steps or metrics), reuse the same reducer:

```python
from group_metrics import reduce_groups
report = reduce_groups(logical_names, groups, compare_arrays, side_a_arrays, side_b_arrays)
```

`groups` maps labels to lists of expected names. Global and group results retain independent coverage; incomplete groups have no aggregate. To quantify differences, pass `lambda a, b: compare_arrays(a, b, metrics=True)` as the reducer. Derive group membership from task coordinates and settings.

For nested state such as model/optimizer/scheduler checkpoints, use the same array arithmetic through [tree_compare.py](../scripts/tree_compare.py):

```python
from tree_compare import compare_trees
report = compare_trees(state_a, state_b, leaf_adapter=export_native_leaf)
```

The adapter converts unsupported framework leaves to NumPy arrays or supported structures, retaining native dtype metadata when export requires conversion. Mapping order is ignored; key/sequence types, metadata and numeric differences remain visible. Select the requested subtrees before comparison. The tool loads no files and imports no execution framework.

Native loaders supply maps of logical names to arrays. `expected` is the task's independent inventory; omit it to check only the observed union. Different native names/layouts can use an explicit adapter mapping. Preserve the observation's input/state, sampling boundary and dtype identity when mapping. Equal names and shapes alone do not establish correspondence.

```sh
python scripts/array_compare.py a.npz b.npz --output report.json
```

The default report contains exact equality, shape/dtype/finite-value checks, counts and bounded findings with native value representations. Error metrics also retain the two values at the maximum-error coordinate. `metrics=True` (CLI `--metrics`) adds the existing FP64 tensor metrics. Report generation is independent of file layout and evidence packaging. Exit 0 means a complete comparison; inspect `exact_equal` for its result.

The [gradient comparison example](../examples/compare_gradients.py) runs native FP32/BF16 backward passes on CPU and directly compares named gradient arrays, retaining actual settings (Torch and NumPy required).

The artifact interfaces below support declared runtime conditions, shard reconstruction and remote observations. Select these capabilities when the task needs them.

Select tools by operation and supported input semantics. The tool library lives in `scripts/`; command-line entrypoints and importable Python functions are both tools. Prepare NumPy for array tools when needed; target-score tools use the standard library.

| Operation | Tool entrypoint | Inputs and outputs |
|---|---|---|
| Inspect tool interfaces | [tool_help.py](../scripts/tool_help.py), CLI | Selected tool names → current module/API docstrings and signatures without imports |
| Compare nested state | [tree_compare.py](../scripts/tree_compare.py), Python `compare_trees` | Native state trees and optional leaf adapter → typed structure/metadata, numeric comparison and coverage |
| Compare corresponding arrays | [array_compare.py](../scripts/array_compare.py), CLI / Python `compare_arrays` | Named arrays or NPZ pair, optional expected names → exact equality, coverage, bounded findings; optional error metrics |
| Continue a multi-stage task | [workflow.py](../scripts/workflow.py), CLI / Python `run_workflow`, `inspect_workflow` | Workload plan and workspace → verified stage reuse, bounded execution, dependency invalidation and next skill action |
| Find reusable prepared resources | [resource_catalog.py](../scripts/resource_catalog.py), CLI / Python `register`, `find` | Declared support properties, resource paths and successful command receipt → content-checked environment/model/adapter/capture lookup |
| Record a bounded native command | [record_run.py](../scripts/record_run.py), CLI / Python `record_run` | Output root, side/task IDs, timeout and literal argv → fresh UTC run with command metadata and streamed stdout/stderr; POSIX timeout cleans up the owned process group |
| Check runtime initialization / interfaces | [runtime_probe.py](../scripts/runtime_probe.py), CLI / Python `probe_runtime`, `inspect_runtime`, `runtime_identity` | Ordered imports, symbols, task checks and optional selected files → actual module/distribution identity, selected source fingerprints and diagnostics; stable identity for workflow reuse |
| Capture fixed-history inference / export scores | [inference_capture.py](../scripts/inference_capture.py), Python `FixedHistory`, `derive_logprobs`, `validate_history`, `validate_generation_scores`, `export_records` | Native full-vocabulary logit row and fixed tokens → declared logprob derivation, history/score-row checks and distribution/target-score records |
| Compare target/action scores | [compare_target_scores.py](../scripts/compare_target_scores.py), CLI | Contract and paired score records → target-score metrics, including EOS and cached execution |
| Compare full distributions | [compare_distributions.py](../scripts/compare_distributions.py), CLI | Contract and paired manifests → top-100 union/tail metrics |
| Compare distributions across hosts | [query_distributions.py](../scripts/query_distributions.py), CLI / Python | Source manifests → inventories and union queries → comparison; full arrays stay at source |
| Aggregate coordinate groups | [group_metrics.py](../scripts/group_metrics.py), Python `reduce_groups` | Expected names/coordinates, named groups, an existing reducer and observation maps → global/per-group metrics and coverage |
| Inspect tensor NaN / Inf | [inspect_nonfinite.py](../scripts/inspect_nonfinite.py), CLI / Python `compare` | Contract and paired tensor manifests → per-side and paired nonfinite counts, bounded logical coordinates and separate finite-subset metrics |
| Compare tensors | [compare_tensors.py](../scripts/compare_tensors.py), CLI / Python `compare`, `Stats` | Contract and paired manifests → observation, gradient, postclip-gradient or stored-update metrics |
| Export logical tensors / compare replica views | [tensor_artifacts.py](../scripts/tensor_artifacts.py), CLI / Python | Per-rank inventories → logical manifests; explicit view pairs → comparisons |
| Compare assistant-body scores | [compare_scores.py](../scripts/compare_scores.py), CLI | Single-turn cache-off score artifacts with execution evidence → metrics |
| Validate artifacts | [validate_artifacts.py](../scripts/validate_artifacts.py), CLI | Contract and one side's tensor/distribution manifest → validation and coverage |
| Package / verify / extract evidence | [evidence_bundle.py](../scripts/evidence_bundle.py), CLI / Python `pack`, `verify`, `extract` | Selected files and optional literal text replacements → indexed source/export hashes and verified extraction |
| Replay retained comparisons | [replay_bundle.py](../scripts/replay_bundle.py), CLI / Python `replay` | Indexed ZIP and `REPLAY.json` → installed comparator/validator reports and expected-value checks |
| Plan msProbe module capture | [msprobe_capture.py](../scripts/msprobe_capture.py), Python `plan_modules` | Native module inventory, selected paths and output sizes → L0 configuration, observation selection and payload estimate; execution uses the [layerwise integration](layerwise.md#integrate-and-verify) |

Supporting modules provide [artifact readers](../scripts/artifact_io.py), [execution-condition checks](../scripts/conditions.py) and [target-score arithmetic](../scripts/score_metrics.py). Reuse them inside tools and task adapters; their module docstrings own the interfaces. Task adapters own native masks, logical mappings, losses and objective reductions.

For manifest-based tools, run from the skill directory. The score, distribution and tensor comparators share `--contract`, `--side-a`, `--side-b` and `--output`; the validator uses `--manifest` for one side's input. Use fresh output paths. For a worked collection task, see the [fixed-history inference example](../examples/fixed-history-inference.md).

```sh
python scripts/compare_tensors.py --contract gradients.json --side-a a.json --side-b b.json --output report.json
```

When preparing manifests, use the selected interface. Distribution matrix rows belong inside each `scores` descriptor, e.g. `scores: {path: rows.npy, row: 3}`; multirow arrays require an explicit row. Put equal conditions in `common` and task-selected differences in `side_conditions`; each rank records its actual settings and imported versions. [Condition checks](../scripts/conditions.py) compare these against the contract-owned rank inventory. For task-selected different initial parameter representations, declare per-side `state`; common `state` retains exact before-value checks. Implement missing task-specific integration and reuse the shared arithmetic.

Distribution/tensor schemas share [artifact_io.py](../scripts/artifact_io.py). Inspect implementation when the selected interface leaves an integration question unresolved.

Contracts set `thresholds: null`. Outputs contain metrics, coordinate counts and unavailable comparisons. Exit codes: 0=comparison metrics computed successfully, 1=unavailable comparison, 2=CLI/output error. Exit 0 does not establish that a user's accuracy tolerance is met.

## Shared tasks

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

For replay, include `REPLAY.json` with named cases selecting a supported validator/comparator, relative input paths and expected report fields. Preserve side identities and order. Replay uses the installed skill CLIs and retained arrays; expected-value checks establish reproducibility of those metrics. The [replay tool docstring](../scripts/replay_bundle.py) owns case fields and supported operations.

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

Use finite target scores/tensors; distribution logprobs permit `-inf` for zero probability. For tensor NaN or Inf, use `inspect_nonfinite.py` with the same contract and manifests. Report anomaly counts and paired finite coverage separately from finite-subset error metrics; matching nonfinite locations do not establish correctness. Its docstring defines supported capture encodings and coordinate limits. Retain the original arrays for repair and evidence replay.
