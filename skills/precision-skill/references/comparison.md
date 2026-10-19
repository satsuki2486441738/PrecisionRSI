# Compare artifacts

Select a tool whose scoring semantics and condition checks match the task. Prepare NumPy for array tools when needed; the target-score comparator uses the standard library.

| Script | Comparison |
|---|---|
| [compare_target_scores.py](../scripts/compare_target_scores.py) | Task-selected target/action logprobs, including EOS and cached execution; standard library |
| [compare_distributions.py](../scripts/compare_distributions.py) | Full-vocabulary logprobs: top-100 union/tail metrics |
| [query_distributions.py](../scripts/query_distributions.py) | Cross-host inventory, union query and comparison; full arrays stay at source |
| [compare_tensors.py](../scripts/compare_tensors.py) | Gradients, postclip gradients and stored before/after updates |
| [tensor_artifacts.py](../scripts/tensor_artifacts.py) | Export logical per-rank inventories, deduplicate equal replicas and compare explicit views |
| [compare_scores.py](../scripts/compare_scores.py) | Single-turn cache-off assistant-body evaluation with execution evidence |
| Task adapter | Native masks, logical mappings, losses and objective reductions; reuse [score arithmetic](../scripts/score_metrics.py) |
| [validate_artifacts.py](../scripts/validate_artifacts.py) | One side's distribution/tensor inventory, values and execution conditions |

Run from the skill directory. The score, distribution and tensor comparators share `--contract`, `--side-a`, `--side-b` and `--output`; the validator uses `--manifest` for one side's input. Use fresh output paths. The [synthetic tensor example](../examples/tensors.py) generates, validates and compares a small pair of artifacts.

```sh
python scripts/compare_tensors.py --contract gradients.json --side-a a.json --side-b b.json --output report.json
```

Read the selected interface before collection. Put equal conditions in `common` and task-selected differences in `side_conditions`; each rank records its actual settings and imported versions. [Condition checks](../scripts/conditions.py) compare these against the contract-owned rank inventory. For task-selected different initial parameter representations, declare per-side `state`; common `state` retains exact before-value checks. Implement missing task-specific integration and reuse the shared arithmetic.

Use `--help` for CLI usage. When adapting inputs, extract only the selected module docstring with Python `ast`; distribution/tensor schemas share [artifact_io.py](../scripts/artifact_io.py). Inspect implementation functions as needed.

Contracts set `thresholds: null`. Outputs contain metrics, coordinate counts and unavailable comparisons. Exit codes: 0=comparison metrics computed successfully, 1=unavailable comparison, 2=CLI/output error.

## Coordinates

Pair scores by sample/prediction position/token and tensors by logical parameter/element. Reconstruct logical coordinates from shards, packed batches and fused projections; retain tied-weight mappings and reduction tags. Check expected coordinates, masks and ranks, and verify converted initial values. Exclude padding, deduplicate equal replicas, and retain differing replicas as explicit views. Distinguish absent gradients from present zeros.

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
