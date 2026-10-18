# Compare artifacts

Select a tool whose scoring semantics and condition checks match the task. Prepare NumPy for array tools when needed; the target-score comparator uses the standard library.

| Script | Comparison |
|---|---|
| [compare_distributions.py](../scripts/compare_distributions.py) | Full-vocabulary logprobs: top-100 union/tail metrics |
| [compare_tensors.py](../scripts/compare_tensors.py) | Gradients or stored before/after updates |
| [compare_scores.py](../scripts/compare_scores.py) | Cache-off single-turn eval, assistant body excluding special tokens: NLL/MAE/P99 |
| Task adapter | Native training scores/objectives or task conditions outside the selected tool interface; reuse metric arithmetic where applicable |
| [validate_artifacts.py](../scripts/validate_artifacts.py) | One side's distribution/tensor inventory and values |

Run from the skill directory. Comparators share `--contract`, `--side-a`, `--side-b` and `--output`; the validator uses `--manifest` for one side's input. Use fresh output paths. The [synthetic tensor example](../examples/tensors.py) generates, validates and compares a small pair of artifacts.

```sh
python scripts/compare_tensors.py --contract gradients.json --side-a a.json --side-b b.json --output report.json
```

Read the selected interface before collection. Its `common` fields enforce equal conditions. Implement or extend a comparator when the task requires differences that this interface cannot express; retain both sides' actual settings and the agreed common conditions in the artifacts.

Use `--help` for CLI usage. When adapting inputs, extract only the selected module docstring with Python `ast`; distribution/tensor schemas share [artifact_io.py](../scripts/artifact_io.py). Inspect implementation functions as needed.

Contracts set `thresholds: null`. Outputs contain metrics, coordinate counts and unavailable comparisons. Exit codes: 0=result, 1=unavailable comparison, 2=CLI/output error.

## Coordinates

Pair scores by sample/prediction position/token and tensors by logical parameter/element. Reconstruct logical coordinates from shards, packed batches and fused projections; retain tied-weight mappings and reduction tags. Check expected coordinates, masks and ranks, and verify converted initial values. Exclude padding, deduplicate equal replicas, and retain differing replicas as explicit views. Distinguish absent gradients from present zeros.

## Target-token scores

For two executions' logprobs $\ell_{a,i},\ell_{b,i}$ over $N$ valid targets, use:

- $\mathrm{NLL}_s=-\sum_i\ell_{s,i}/N$; report each side's NLL and their absolute difference.
- $e_i=|\ell_{a,i}-\ell_{b,i}|$; report MAE, P99 and maximum with its coordinate.

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
