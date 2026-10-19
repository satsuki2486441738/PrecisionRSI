# Compare artifacts

## Tools

Select tools by operation and supported input semantics. The tool library lives in `scripts/`; command-line entrypoints and importable Python functions are both tools. Prepare NumPy for array tools when needed; target-score tools use the standard library.

| Operation | Tool entrypoint | Inputs and outputs |
|---|---|---|
| Check runtime initialization / interfaces | [runtime_probe.py](../scripts/runtime_probe.py), CLI / Python `probe_runtime`, `inspect_runtime` | Ordered imports, symbols and task checks → actual module identity, distribution metadata and diagnostics in a fresh interpreter or native worker |
| Capture fixed-history inference / export scores | [inference_capture.py](../scripts/inference_capture.py), Python `FixedHistory`, `validate_history`, `export_records` | Native full-vocabulary logit row and fixed tokens → preconstraint logprob arrays, history checks and distribution/target-score records |
| Compare target/action scores | [compare_target_scores.py](../scripts/compare_target_scores.py), CLI | Contract and paired score records → target-score metrics, including EOS and cached execution |
| Compare full distributions | [compare_distributions.py](../scripts/compare_distributions.py), CLI | Contract and paired manifests → top-100 union/tail metrics |
| Compare distributions across hosts | [query_distributions.py](../scripts/query_distributions.py), CLI / Python | Source manifests → inventories and union queries → comparison; full arrays stay at source |
| Compare tensors | [compare_tensors.py](../scripts/compare_tensors.py), CLI | Contract and paired manifests → observation, gradient, postclip-gradient or stored-update metrics |
| Export logical tensors / compare replica views | [tensor_artifacts.py](../scripts/tensor_artifacts.py), CLI / Python | Per-rank inventories → logical manifests; explicit view pairs → comparisons |
| Compare assistant-body scores | [compare_scores.py](../scripts/compare_scores.py), CLI | Single-turn cache-off score artifacts with execution evidence → metrics |
| Validate artifacts | [validate_artifacts.py](../scripts/validate_artifacts.py), CLI | Contract and one side's tensor/distribution manifest → validation and coverage |
| Plan msProbe module capture | [msprobe_capture.py](../scripts/msprobe_capture.py), Python `plan_modules` | Native module inventory, selected paths and output sizes → L0 configuration, observation selection and payload estimate; execution uses the [layerwise integration](layerwise.md#integrate-and-verify) |

Supporting modules provide [artifact readers](../scripts/artifact_io.py), [execution-condition checks](../scripts/conditions.py) and [target-score arithmetic](../scripts/score_metrics.py). Reuse them inside tools and task adapters; their module docstrings own the interfaces. Task adapters own native masks, logical mappings, losses and objective reductions.

Run from the skill directory. The score, distribution and tensor comparators share `--contract`, `--side-a`, `--side-b` and `--output`; the validator uses `--manifest` for one side's input. Use fresh output paths. The [synthetic tensor example](../examples/tensors.py) generates, validates and compares a small pair of artifacts. For a worked collection task, see the [fixed-history inference example](../examples/fixed-history-inference.md).

```sh
python scripts/compare_tensors.py --contract gradients.json --side-a a.json --side-b b.json --output report.json
```

Read the selected interface before collection. Distribution matrix rows belong inside each `scores` descriptor, e.g. `scores: {path: rows.npy, row: 3}`; multirow arrays require an explicit row. Put equal conditions in `common` and task-selected differences in `side_conditions`; each rank records its actual settings and imported versions. [Condition checks](../scripts/conditions.py) compare these against the contract-owned rank inventory. For task-selected different initial parameter representations, declare per-side `state`; common `state` retains exact before-value checks. Implement missing task-specific integration and reuse the shared arithmetic.

Use `--help` for CLI usage. When adapting inputs, extract only the selected module docstring with Python `ast`; distribution/tensor schemas share [artifact_io.py](../scripts/artifact_io.py). Inspect implementation functions as needed.

Contracts set `thresholds: null`. Outputs contain metrics, coordinate counts and unavailable comparisons. Exit codes: 0=comparison metrics computed successfully, 1=unavailable comparison, 2=CLI/output error.

## Coordinates

Pair scores by sample/prediction position/token and tensors by logical tensor/element. For `stage: observations`, declare each tensor's logical sampling `coordinates` in the contract and both manifests; retain sample/token selection, boundary, invocation and execution stage as applicable. Native names map to this identity in the adapter. Reconstruct logical coordinates from shards, packed batches and fused projections; retain tied-weight mappings and reduction tags. Check expected coordinates, masks and ranks, and verify converted initial values. Exclude padding, deduplicate equal replicas, and retain differing replicas as explicit views. Distinguish absent gradients from present zeros.

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
| Maximum absolute error | $\max_i|a_i-b_i|$ |

Global metrics concatenate coordinates. Retain counts and signal/error norms. Actual updates are $\theta_{after}-\theta_{before}$, promoting stored values to at least FP32 before subtraction.

Cosine is undefined when either norm is zero. Represent undefined metrics as null with a reason.

Use finite target scores/tensors; distribution logprobs permit `-inf` for zero probability.
