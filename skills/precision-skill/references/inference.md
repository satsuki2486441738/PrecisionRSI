# Inference

For vLLM or SGLang, also read [serving engines](engines.md) for native prefill/decode collection. The cache-off full-forward procedure below is a separate scope, not an instruction to replace an engine's execution path.

## Common forward

Use the same model/weights/tokenizer/config, fixed token history, position IDs, attention mask and selected prediction positions. Preserve serialization and complete samples; reuse existing selection/tokenization records. Match parameter dtype, quantization and autocast policy while retaining each platform's native kernels.

Run full-forward in eval mode with autograd disabled and cache off. At prediction position $t-1$, compare the next-token distribution for the same prefix. Compute full-vocabulary natural-log softmax in FP32 from converted logits, before sampling processors or temperature. For assistant-body scoring, select nonspecial body positions; labels are needed only for auxiliary target scoring.

## Top-100 distribution comparison

1. At each agreed position, each side selects its 100 highest-scoring token IDs; break exact score ties by ascending ID. Let these sets be $T_G$ and $T_N$; form the union $S=T_G\cup T_N$ (100–200 IDs).
2. Obtain **both sides' original full-vocabulary-normalized probabilities for every ID in $S$**. Retain full score arrays for offline queries, or use an authorized same-state replay. Two independent top-100 files alone are insufficient; missing scores are not zero.
3. Preserve each side's top-100 probability mass and the mass outside $S$. Combine that tail into one `other` category. Do not renormalize only the selected tokens.
4. Pair by sample and prediction position, verify the complete history and position set, then compute the following metrics with GPU probabilities $p$ and NPU probabilities $q$.

| Metric | Per-position definition | Across positions |
|---|---|---|
| Tail-merged KL | $\sum_{i\in S\cup\{\mathrm{other}\}}p_i\log\frac{p_i}{q_i}$ | Mean, P99, maximum with coordinate. |
| Top-1 agreement | Equal argmax token IDs. | Agreement fraction and count. |
| Top-100 overlap | $\frac{\lvert T_G\cap T_N\rvert}{100}$ | Mean. |
| Probability RMSE | $\sqrt{\frac{1}{\lvert S\rvert}\sum_{i\in S}(q_i-p_i)^2}$ | Mean, P99, maximum. |
| Maximum probability error | $\max_{i\in S}\lvert q_i-p_i\rvert$ | Mean, P99, maximum with position/token ID. |
| Coverage | Each side's own top-100 mass and union mass. | Mean, P1, minimum, separately for each side. |

Weight positions equally; calculate RMSE per position before aggregation. Merge positions across samples/batches/ranks before linear percentiles. Retain top-1 mismatch examples with both candidates' probabilities and the reference first/second probability margin.

Use stable tail accumulation and FP64 offline metric arithmetic. Record any full-vocabulary numerical renormalization and its serialization tolerance in the case; this is not an acceptance threshold. Zero reference mass contributes zero to KL; positive reference mass with zero candidate mass gives infinite divergence. Preserve logprobs to distinguish underflow from true zero.

If KL is nonfinite, retain the affected coordinate and score evidence; serialize its value as null with an explicit reason and report `incomparable`. Do not drop that position or present aggregates over only the remaining positions as complete coverage.

Tail-merged KL is a lower bound on full-vocabulary KL: it cannot detect redistribution within the tail. Coverage is therefore part of the result. Auxiliary labeled-token NLL and errors use [comparison](comparison.md); they do not replace distribution metrics.

## Generation and cache

Free generation and cached decoding are separate scopes. After histories diverge, equal token indices are no longer common-input observations. For cache comparison, distinguish evolving native caches along a fixed history from a one-step comparison with an identical injected cache. Neither is covered by full-forward results.
