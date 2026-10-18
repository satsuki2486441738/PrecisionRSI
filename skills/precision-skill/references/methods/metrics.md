# Metric definitions

## Coordinates

Pair scores by sample/prediction position/token and tensors by logical parameter/element. Check expected coordinates, masks and ranks. Exclude padding, deduplicate equal replicas, and retain differing replicas as explicit views. Distinguish absent gradients from present zeros.

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

Use finite target scores/tensors; distribution logprobs permit `-inf` for zero probability. Compare user-supplied tolerances to the computed metric and aggregation.

## Distributions

1. Select each side's top 100 token IDs at every position, breaking exact ties by ascending token ID.
2. Form their union $S$ and obtain both sides' original full-vocabulary-normalized probabilities for every ID in $S$.
3. Accumulate probability outside $S$ into one `other` category. Retain each side's own top-100 mass and union mass.
4. Compute the following with the two executions' probabilities $p$ and $q$.

| Metric | Per position | Aggregate |
|---|---|---|
| Tail-merged Jensen–Shannon divergence | $\frac12\sum_i[p_i\log(p_i/m_i)+q_i\log(q_i/m_i)]$, $m=(p+q)/2$, $i\in S\cup\{other\}$ | Mean, P99, maximum and coordinate |
| Top-1 agreement | Equal argmax token IDs | Fraction and count |
| Probability RMSE | $\sqrt{\sum_{i\in S}(q_i-p_i)^2/|S|}$ | Mean, P99, maximum |
| Maximum probability error | $\max_{i\in S}|q_i-p_i|$ | Mean, P99, maximum and token |
| Probability-mass coverage | Own top-100 and union mass for each side | Mean, P1, minimum |
| Secondary top-100 overlap | $|T_A\cap T_B|/100$ | Mean |

Weight positions equally and use FP64 offline arithmetic. Compute RMSE per position before aggregation. Retain top-1 mismatch examples with both token probabilities and each side's first/second margin.

Use stable complement accumulation and a log-space mixture for Jensen–Shannon divergence in nats. Zero probability contributes zero. Tail merging compares the union plus one complement category.
