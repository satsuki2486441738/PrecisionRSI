# Inference

## Collect scores

Replay identical token IDs, position IDs, attention masks and prediction positions using the same checkpoint/tokenizer/config and numerical policy. For cache-off full forward, use eval/no-grad. At prediction position $t-1$, collect the distribution predicting token $x_t$.

Compute full-vocabulary natural-log softmax in FP32 before temperature or sampling processors, retaining model-defined scaling/softcapping. Save full logprob rows or queryable arrays, sample/position coordinates and vocabulary mapping.

Feed the same predetermined continuation while each side evolves its native cache. Record prefill/decode stages, batching, cache settings and context length with the scores.

Collect requested label logprobs even when labels lie outside top-100. Use the offline [tools](../methods/tools.md) for distribution and target-score comparison. Group results by prefill/first decode/later decode and requested context-length bins. For generated continuations, pair scores while token histories match.

## Framework integration

Record attention backend, TP, KV dtype, graph/cache/chunking and batching settings. Map reordered requests/chunks to absolute prediction positions: prompt token $x_t$ uses position $t-1$; the final prefill row supplies the first output distribution. Account for BOS/templates and initial prompt entries.

Reconstruct vocabulary shards, exclude padding and export the score rows needed for union queries. Preserve each engine's native cache while replaying a predetermined continuation.

### vLLM

Inspect `SamplingParams.prompt_logprobs`, `logprobs`, `max_logprobs` and the platform plugin. Check full-vocabulary support such as `-1` and `logprobs_mode="raw_logprobs"` in the installed release. Handle initial `None` prompt entries and chunk offsets. Use engine replay or a collector for fixed-history decode.

### SGLang

Inspect `return_logprob`, `logprob_start_len`, `top_logprobs_num` and `token_ids_logprob`. Set prompt coverage explicitly and check selected-ID queries at input/output positions. Trace temperature, penalties and constraints to select raw scores; capture native logits when needed. Preserve radix-cache and extend/decode behavior.

### Other/custom

Follow [native adaptation](../methods/collection.md#adapt-an-unfamiliar-implementation). Locate the model or engine's native scoring path, resolve score transforms and map outputs to token/vocabulary coordinates. For cached execution, replay the fixed continuation through its native cache and identify prefill/decode boundaries.

## Metrics

| Observations | Report |
|---|---|
| Output distributions | Tail-merged Jensen–Shannon divergence, top-1 agreement, probability RMSE and maximum probability error |
| Requested target scores | Each side's NLL and absolute NLL difference; logprob MAE, P99 and maximum |
| Coverage | Paired positions/targets, each side's top-100 and union mass; secondary top-100 overlap |

Use the [distribution definitions](../methods/metrics.md#distributions) and [target-score definitions](../methods/metrics.md#target-token-scores). Keep the prefill/decode and context-length breakdowns requested by the task.
