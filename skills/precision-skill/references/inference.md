# Inference

## Collect scores

Replay identical token IDs, position IDs, attention masks and prediction positions using the same checkpoint/tokenizer/config and numerical policy. For cache-off full forward, use eval/no-grad. At prediction position $t-1$, collect the distribution predicting token $x_t$.

Compute full-vocabulary natural-log softmax in FP32 before temperature or sampling processors, retaining model-defined scaling/softcapping. Save full logprob rows or queryable arrays, sample/position coordinates and vocabulary mapping.

When adapting native engine collection, read the selected [vLLM / SGLang](engines.md) section. Feed the same predetermined continuation while each side evolves its native cache. Record prefill/decode stages, batching, cache settings and context length with the scores.

## Top-100 distribution comparison

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

## Target scores and stage breakdowns

Collect requested label logprobs even when labels lie outside top-100. Use [target-score metrics](comparison.md#target-token-scores) for NLL and score errors. Group engine results by prefill/first decode/later decode and the requested context-length bins. For generated continuations, pair scores while token histories match.
