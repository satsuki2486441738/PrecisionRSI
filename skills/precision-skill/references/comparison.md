# Validation and metrics

## Comparable observations

Check common model/data identity, numerical policy, input history, task state and observation stage before computing paired differences. Verify actual execution evidence and completion for every expected rank. Preserve platform-specific kernels/settings separately; shared implementation alignment follows [environment](environment.md).

Pair by semantic coordinates: sample/position/token for scores, logical parameter/element for tensors. Validate lengths, masks, uniqueness and complete required coverage. Reordering is allowed; silently intersecting sets, replacing missing values with zero or dropping nonfinite values is not. Layout transformations must follow an independently established mapping, not one fitted to minimize error.

Separate declared settings, execution evidence and unknowns. Required unknowns block only the affected scope; unneeded internal details remain coverage limits.

## Target-token scores

For $N$ paired valid targets, let $\ell_{G,i}$ and $\ell_{N,i}$ be their GPU and NPU natural log probabilities. Require finite scores at most zero, including zero as valid.

$$
\mathrm{NLL}_{s}=-\frac{1}{N}\sum_{i=1}^{N}\ell_{s,i},\qquad
\Delta\mathrm{NLL}=\mathrm{NLL}_{N}-\mathrm{NLL}_{G},\qquad s\in\{G,N\}.
$$

- $e_i=\lvert\ell_{N,i}-\ell_{G,i}\rvert$; $\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}e_i$; also report maximum error when requested.
- Sum over all samples/batches/ranks before dividing by valid-target count. Signed delta NLL can cancel local errors, so keep absolute-error metrics.
- For a linear percentile $a$, sort all errors, set $h=a(N-1)$, and interpolate between zero-based indices $\lfloor h\rfloor$ and $\lceil h\rceil$. At $N=1$ the percentile is the sole value. P99 uses $a=0.99$; do not average per-rank percentiles.

Next-token distribution metrics are defined in [inference](inference.md#top-100-distribution-comparison).

## Tensors and actual updates

For aligned reference $r$ and candidate $c$ with $N$ coordinates, compute in FP64 with stable reductions:

| Metric | Definition | Zero-norm handling |
|---|---|---|
| Relative L2 | $\frac{\lVert c-r\rVert_2}{\lVert r\rVert_2}$ | Both reference and error norms zero: 0. Zero reference with nonzero error: undefined. |
| Cosine | $\frac{r^\top c}{\lVert r\rVert_2\lVert c\rVert_2}$ | Undefined if either norm is zero, even for two equal zero vectors. |
| Maximum absolute error | $\max_i\lvert c_i-r_i\rvert$ | Defined for nonempty finite coordinates. |
| Optional RMSE | $\sqrt{\frac{1}{N}\sum_{i=1}^{N}(c_i-r_i)^2}$ | Defined for nonempty finite coordinates. |

Do not add epsilon to relative-L2 denominators. Report coordinate counts, both signal norms, error norm and undefined metrics explicitly. Global metrics use concatenated coordinates, not averages of per-tensor ratios. For updates, compare $\Delta\theta=\theta_{\mathrm{after}}-\theta_{\mathrm{before}}$ from each side, not final parameters alone. [Training](training.md) specifies collection stages and coverage.

Separate execution, comparability and metric results. Encode undefined zero-norm metrics as null with a reason and metric-level `incomparable`; retain defined errors as `no_thresholds` without thresholds. Undefined cosine does not invalidate other metrics. Missing/nonfinite observations block their affected scope.

## Interpretation

Thresholds, when requested, apply to a specified metric, reference, coverage and aggregation and must be fixed before the target comparison. Do not fit thresholds to observed discrepancies. Without thresholds, use `no_thresholds`; self-comparison only checks the pipeline.

| Status | Meaning |
|---|---|
| passed / exceeded | Complete comparable observations meet / violate predefined thresholds. |
| no_thresholds | Comparable diagnostic without predefined acceptance thresholds. |
| incomparable | Conflicting semantics/state/mapping or an undefined/nonfinite metric domain. |
| not_covered | Missing required observations, execution or capability. |
| execution_failed | Collection or computation failed. |

Retain all blockers and usable single-side diagnostics. A concise result should state measured scope and limitations; successful execution is not a precision pass.

## Bundled target-score comparator

[compare_scores.py](../scripts/compare_scores.py) supports schema-v2 full-forward **target-token** scoring with null thresholds, as described in [SKILL.md](../SKILL.md#available-tool). Other scopes need case tooling.

Use the module docstring for the schema. Paths are relative to their containing JSON; `actual` fields must reference execution records. The agent reviews provenance/version alignment, which the numeric core cannot establish.

```bash
python scripts/compare_scores.py --contract <contract.json> --reference <gpu.json> --candidate <npu.json> --output <new-report.json>
```

Run from the skill directory. Output creation is exclusive; exit codes are 0 for a diagnostic, 1 for a written blocked report, and 2 for argument/output errors. Required unsupported coverage remains explicit rather than being reinterpreted as target scoring.
