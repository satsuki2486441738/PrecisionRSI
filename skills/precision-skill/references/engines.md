# vLLM and SGLang

Use the [inference](inference.md) top-100 distribution metrics for each engine's GPU/NPU pair. Comparing vLLM against SGLang is a distinct experiment. These are collection methods, not bundled engine adapters; implement and validate the collector against the installed release and platform backend.

## Native execution and coordinates

Reuse the case's engine/backend versions, model, tokenized inputs and numerical policy. Record the actual engine commit, platform plugin/patches (including vLLM-Ascend where used), attention backend, TP/PP/DP/EP layout, KV-cache dtype, graph mode, prefix-cache policy, chunked-prefill settings and scheduler/batch conditions that affect the selected path. Use compatible approved GPU/NPU version pairings; native kernels need not match.

Separate native prefill from fixed-history decode; neither establishes cache-off full-forward agreement. For decode, feed the same predetermined continuation at each step and observe scores before choosing the next token. Each side evolves its own native cache along that history. Re-prefilling every prefix only tests prefill; identical injected-cache comparison is another scope. Free generation can compare distributions only while complete histories remain equal.

Pair rows by stable sample ID, full input history and absolute prediction position, not response order, worker rank, chunk index or decoded text. A prompt score for token x[t] belongs to the distribution after x[:t], at prediction position t-1. The initial token generally has no preceding score; the final prompt row predicts the first output token. Verify these boundaries, BOS/chat-template insertion, chunk transitions and first decode step with a tiny known-token request before collection. Save the expected row inventory; omitted/cached/truncated rows are missing coverage, not zero scores.

Preserve the agreed cache, batching and parallel execution path. If cache hits prevent required observations, resolve collection capability before the run; disabling cache or changing batching is a separately recorded diagnostic variant. For vocabulary parallelism, reconstruct logical vocabulary IDs and global normalization, excluding padded vocabulary entries; never normalize each shard independently or count replicas twice.

## Observation boundary

Collect full-vocabulary FP32 natural-log softmax of the model's logits before sampling temperature, penalties, constraints, top-k/top-p or other sampling processors. Preserve model-defined logit scaling/softcapping. Inspect the actual backend path: a field named `logprob` or a greedy request does not prove this boundary. Prefer a verified native API; otherwise capture at the engine's logits boundary without substituting a Transformers forward or disabling native attention/fusion.

For a single run per side with offline exchange, retain full normalized score rows (or full raw logits sufficient to derive them). Two top-100 responses cannot score their union. Authorized replay or verified selected-token queries can supply union scores only under the same history, state and execution conditions. Validate API limits, score dtype/normalization and complete positions before expensive collection. Sort top-100 by score then token ID; an API may include an extra sampled token or resolve cutoff ties differently, so response length alone does not define the set.

## vLLM

- Inspect `SamplingParams.prompt_logprobs` for prompt rows and `logprobs` for output rows, plus the engine's `max_logprobs` limit. Releases supporting `-1` can expose full-vocabulary scores; verify this on both platform builds and the chosen endpoint rather than assuming the OpenAI-compatible API has identical capabilities.
- Where available, `logprobs_mode="raw_logprobs"` requests pre-processor probabilities; raw-logit modes require full-vocabulary FP32 log-softmax. Processed scores do not satisfy the raw-distribution contract. Confirm the runner/sampler and platform plugin actually implement the selected mode.
- Prompt arrays may begin with `None`; returned dictionaries can include the observed token in addition to top-k. Explicitly map prompt and output rows, including chunked prefill. Ordinary generation does not force a shared decode history: use a supported replay facility or a case collector that feeds fixed tokens while retaining native cache execution.

## SGLang

- Inspect native `return_logprob`, `logprob_start_len`, `top_logprobs_num` and `token_ids_logprob` support. Set the prompt start explicitly for the agreed coverage; output-only defaults do not collect prompt distributions. Map input/output logprob arrays by token IDs and causal positions, including any initial placeholder and extend/chunk offsets.
- Check the installed logits processor and sampler for input versus output score semantics. Output logprobs can be temperature-adjusted; releases with `SGLANG_RETURN_ORIGINAL_LOGPROB` still require checking where penalties/constraints occur. A setting alone is not evidence of raw scores. Capture before processors when the API cannot meet the observation boundary.
- Top-k and selected-ID responses are partial views, not a full-distribution export. Verify the actual limits and whether requested IDs cover prompt as well as output rows. For one-run offline exchange, collect full rows inside the native engine when the endpoint cannot return them. Preserve radix/prefix-cache and extend/decode behavior; a score-only prefill request does not validate decode.

## Diagnose and report

Validate raw-score semantics, row/vocabulary coverage and common history before applying [inference metrics](inference.md#top-100-distribution-comparison). First localize a mismatch to prefill, a chunk boundary or a decode step; then inspect the smallest relevant evidence for position mapping, cache state, attention, parallel reductions or sampling contamination. Report engine and stage coverage, metrics and remaining limitations. CPU adapter tests or successful requests do not establish GPU/NPU numerical agreement.
