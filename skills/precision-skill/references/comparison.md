# Metric definitions

## Coordinates

Pair scores by sample/prediction position/token and tensors by logical parameter/element. Check expected coordinates, masks and ranks. Exclude padding, deduplicate equal replicas, and retain differing replicas as explicit views. Distinguish absent gradients from present zeros.

## Target-token scores

For GPU/NPU logprobs $\ell_{G,i},\ell_{N,i}$ over $N$ valid targets, use:

- $\mathrm{NLL}_s=-\sum_i\ell_{s,i}/N$; signed difference is NPU minus GPU.
- $e_i=|\ell_{N,i}-\ell_{G,i}|$; report MAE, P99 and maximum with its coordinate.

Use the global valid-target count. Native loss retains its implemented reduction. Percentiles use all paired errors, linearly interpolated at $a(N-1)$; P99 uses $a=0.99$.

Distribution metrics are defined in [inference](inference.md#top-100-distribution-comparison).

## Tensors

For reference $r$ and candidate $c$, use stable FP64 reductions:

| Metric | Definition |
|---|---|
| Relative L2 | $\lVert c-r\rVert_2/\lVert r\rVert_2$ |
| Cosine | $r^\top c/(\lVert r\rVert_2\lVert c\rVert_2)$ |
| Maximum absolute error | $\max_i|c_i-r_i|$ |

Global metrics concatenate coordinates. Retain counts and signal/error norms. Actual updates are $\theta_{after}-\theta_{before}$, promoting stored values to at least FP32 before subtraction.

Relative L2 is zero when reference and error norms are zero, and undefined for zero reference with nonzero error. Cosine is undefined when either norm is zero. Represent undefined metrics as null with a reason.

Use finite target scores/tensors; distribution logprobs permit `-inf` for zero probability. Compare user-supplied tolerances to the computed metric and aggregation.
