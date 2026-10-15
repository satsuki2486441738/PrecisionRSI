# vLLM and SGLang

Apply [inference](inference.md) metrics to each engine's GPU/NPU pair. Inspect the installed release and platform backend before choosing an API or hook; use its native execution path.

## Native collection

- Record effective attention backend, parallel layout, KV dtype, graph mode, prefix-cache/chunking policy and batching conditions. Preserve these during collection; changed paths are diagnostic variants.
- Separate prefill from fixed-history decode. Feed the same predetermined continuation while each side evolves its native cache; re-prefilling prefixes does not test decode. Free-generation distributions are comparable only while histories match.
- Map sample/history and absolute prediction position through chunks and request reordering. A score for prompt token $x_t$ belongs to position $t-1$; the initial token usually has no score, and the first output distribution belongs to the final prefill row. Verify BOS/template insertion and these boundaries within the authorized run budget; retain the expected position inventory.
- Collect FP32 full-vocabulary log-softmax before sampling processors, retaining model-defined scaling/softcapping. Verify API semantics; otherwise hook native logits. Reconstruct global vocabulary normalization across shards, exclude padding and deduplicate replicas.
- Follow inference's union-score requirements. One run per side with offline exchange requires full score rows or full raw logits. Partial queries/replay must preserve history, state and execution conditions. API top-k may include an extra sampled token or resolve ties differently; apply the protocol's deterministic selection.

## vLLM

Check `SamplingParams.prompt_logprobs`, `logprobs` and engine `max_logprobs`. Some releases support `-1` for full vocabulary and `logprobs_mode="raw_logprobs"`; verify the chosen endpoint and platform plugin. Raw logits need full-vocabulary normalization; processed scores do not meet the observation boundary. Handle initial `None` prompt entries and chunk offsets. Ordinary generation needs a replay facility or collector to enforce fixed-history decode.

## SGLang

Check `return_logprob`, `logprob_start_len`, `top_logprobs_num` and `token_ids_logprob`. Set prompt coverage explicitly; output-only defaults omit prompt rows. Verify selected-ID support for both input and output positions. Output scores may be temperature-adjusted; even `SGLANG_RETURN_ORIGINAL_LOGPROB` requires inspecting preceding penalties/constraints. Capture native logits if the endpoint cannot export complete raw distributions. Preserve radix/prefix-cache and extend/decode behavior.

## Reporting and localization

Use existing metrics, split by prefill, first cached decode and later decode, and separately by a few preselected context-length bins. Context length is the full history producing a row; use common bins and retain counts or explicit missing coverage. Keep detailed breakdowns in the evidence.

Localize to a position/stage before inspecting cache, attention or reductions. Only for an observed anomaly, vary one condition: cached decode versus re-prefill, prefix-cache hit versus miss, chunked versus unchunked prefill, or single versus batched requests. Compare common histories and report variants separately within budget. Internal-tensor and free-generation metrics are not default additions.
