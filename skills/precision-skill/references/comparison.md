# Validation and metrics

## Comparable observations

Check common model/data identity, numerical policy, input history, task state and observation stage before computing paired differences. Verify actual execution evidence and completion for every expected rank. Preserve platform-specific kernels/settings separately; shared implementation alignment follows [environment](environment.md).

Pair by semantic coordinates: sample/position/token for scores, logical parameter/element for tensors. Validate lengths, masks, uniqueness and complete required coverage. Reordering is allowed; silently intersecting sets, replacing missing values with zero or dropping nonfinite values is not. Layout transformations must follow an independently established mapping, not one fitted to minimize error.

Keep declared settings, runtime evidence and unknowns distinct. Look up existing evidence before recollecting. Required unknowns block the affected scope; unneeded internal details remain coverage limits. Parameter dtype or FP32 output does not establish every internal operation's dtype.

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

Training reports separate execution and input/state/coverage comparability from per-metric results; no single overall numerical verdict is required. Encode a zero-norm metric that is undefined as null with its reason and metric-level `incomparable`; retain defined errors as `no_thresholds` when no thresholds exist. An undefined cosine alone does not invalidate other metrics or imply a state/coverage mismatch. Nonfinite observations and missing coordinates still block their affected comparison scope.

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

[compare_scores.py](../scripts/compare_scores.py) is standard-library-only, schema v2 / core 2.0.0. It validates full-forward **target-token scores** and computes NLL, delta NLL, MAE and global linear P99. It supports only null thresholds and does not emit passed/exceeded. Top-100, training and cache observations require separate case tooling.

Use the module's opening docstring as the authoritative field/type contract instead of duplicating its schema. Paths resolve relative to their containing JSON. `actual` fields need pointers to matching execution records; the contract cannot certify itself. The coordinator reviews provenance and shared versions, which the numeric core cannot establish automatically.

```bash
python scripts/compare_scores.py --contract <contract.json> --reference <gpu.json> --candidate <npu.json> --output <new-report.json>
```

Run from the skill directory. Output creation is exclusive; exit codes are 0 for a diagnostic, 1 for a written blocked report, and 2 for argument/output errors. Required unsupported coverage remains explicit rather than being reinterpreted as target scoring.
