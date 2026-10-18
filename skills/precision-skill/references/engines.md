# Serving-engine capture

Use the [inference protocol](inference.md). Read the selected engine section.

## Common mapping

Record attention backend, TP, KV dtype, graph/cache/chunking and batching settings. Map reordered requests/chunks to absolute prediction positions: prompt token $x_t$ uses position $t-1$; the final prefill row supplies the first output distribution. Account for BOS/templates and initial prompt entries.

Reconstruct vocabulary shards, exclude padding and export the score rows needed for union queries. Preserve each engine's native cache while replaying a predetermined continuation.

## vLLM

Inspect `SamplingParams.prompt_logprobs`, `logprobs`, `max_logprobs` and the platform plugin. Check full-vocabulary support such as `-1` and `logprobs_mode="raw_logprobs"` in the installed release. Handle initial `None` prompt entries and chunk offsets. Use engine replay or a collector for fixed-history decode.

## SGLang

Inspect `return_logprob`, `logprob_start_len`, `top_logprobs_num` and `token_ids_logprob`. Set prompt coverage explicitly and check selected-ID queries at input/output positions. Trace temperature, penalties and constraints to select raw scores; capture native logits when needed. Preserve radix-cache and extend/decode behavior.
