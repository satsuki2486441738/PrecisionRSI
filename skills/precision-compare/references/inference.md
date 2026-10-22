# Inference

## Collection

Replay the selected token IDs, position IDs, attention masks and prediction positions. Check checkpoint/tokenizer/config and numerical settings against the task's common conditions and per-side configuration. At prediction position $t-1$, collect the distribution predicting token $x_t$.

Collect the observations needed by the selected metrics. For target-score comparison, save the requested labels, logical positions and full-vocabulary-normalized logprobs. For distribution comparison, use full-vocabulary-normalized scores before sampling processors, retaining model-defined scaling/softcapping. When deriving logprobs from logits, declare the FP32/FP64 policy and record native score dtype separately from derivation dtype; save full logprob rows or queryable arrays and vocabulary mapping. Retain requested label scores even outside top-100.

### Full forward

For cache-off scoring, use eval/no-grad and collect the requested prediction positions in the native forward pass. Record actual cache flags and forward shapes.

### Cached execution

Feed the same predetermined continuation while each side evolves its native cache. Record prefill/decode stages, batching, cache settings and context length. For generated continuations, pair scores while token histories match.

## Framework integration

Use the relevant entrypoint below. Inspect the installed implementation when native score semantics or coordinate mappings are unresolved.

Record applicable attention backend, TP, KV and recurrent-state dtypes, graph/cache/chunking and batching settings. Map reordered requests/chunks to absolute prediction positions: prompt token $x_t$ uses position $t-1$; the final prefill row supplies the first output distribution. Account for BOS/templates and initial prompt entries.

Reconstruct vocabulary shards, exclude padding and export the score rows needed for union queries. Preserve each engine's native cache while replaying a predetermined continuation.

| Framework / entrypoint | Inspect in the installed implementation |
|---|---|
| PyTorch / [Transformers](https://huggingface.co/docs/transformers/main_classes/text_generation) | Native model forward and cache state; distinguish raw logits from processed generation scores |
| [vLLM](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/) | `prompt_logprobs`, output `logprobs`, raw-score mode and full-vocabulary limits; resolve initial `None` entries, chunk offsets and platform plugins |
| [SGLang](https://docs.sglang.io/docs/basic_usage/sampling_params) | `return_logprob`, `logprob_start_len`, top/selected-ID queries and score transforms; preserve radix-cache and extend/decode behavior |
| [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/features/additional-outputs.html) | Installed execution backend, context/generation logits, vocabulary layout and effective build/precision settings |
| [LMDeploy](https://lmdeploy.readthedocs.io/en/latest/) | Selected PyTorch/TurboMind backend, native score outputs and request/token/cache mappings |
| [MindIE-LLM](https://github.com/Ascend/MindIE-LLM) | Service/model backend, available score access and scheduled request/prefill/decode mappings |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF/checkpoint identity, quantization, tokenizer conversion and native context/logit-row mapping |

Resolve score availability at the native execution layer and export the observations required by the selected metrics. Use engine replay or a collector when fixed-history decode is requested. The [inference capture tool](../scripts/inference_capture.py) saves full-vocabulary logprobs before forcing each fixed token, checks logical prefill/decode histories and exports distribution/target-score records. FP32 remains the default; `logprob_dtype='float64'` explicitly derives logprobs offline from copied native values. Use `validate_generation_scores` to check context/generation row alignment when both outputs are available. `export_records` verifies normalization and target scores against the selected saved rows. The adapter selects request rows, reconstructs vocabulary shards and maps native scheduling events to logical events.

### Other/custom

Follow [native adaptation](../SKILL.md#adapt-an-unfamiliar-implementation). Locate the model or engine's native scoring path, resolve score transforms and map outputs to token/vocabulary coordinates. For cached execution, replay the fixed continuation through its native cache and identify prefill/decode boundaries.

## Metrics

| Observations | Report |
|---|---|
| Output distributions | Tail-merged Jensen–Shannon divergence, top-1 agreement, probability RMSE and maximum probability error |
| Requested target scores | Each side's NLL and absolute NLL difference; logprob MAE, P99 and maximum |
| Coverage | Paired positions/targets, each side's top-100 and union mass; secondary top-100 overlap |

Use [comparison](comparison.md) for target-score metrics and tool usage. Across hosts, use its distribution-query tool to exchange top-100 inventories, union scores and exact tail statistics while keeping full arrays at source. Keep the prefill/first-decode/later-decode and context-length breakdowns requested by the task.

### Distribution comparison

1. Select each side's top 100 token IDs at every position, breaking exact ties by ascending token ID.
2. Form their union $S$ and obtain both sides' original full-vocabulary-normalized probabilities for every ID in $S$.
3. Accumulate probability outside $S$ into one `other` category. Retain each side's own top-100 mass and union mass.
4. Compute the following with the two executions' probabilities $p$ and $q$.

| Metric | Per position | Aggregate |
|---|---|---|
| Tail-merged Jensen–Shannon divergence | $\frac12\sum_i[p_i\log(p_i/m_i)+q_i\log(q_i/m_i)]$, $m=(p+q)/2$, $i\in S\cup\{other\}$ | Mean, P99, maximum and coordinate |
| Top-1 agreement | Equal argmax token IDs | Fraction and count |
| Probability RMSE | $\sqrt{\sum_{i\in S}(q_i-p_i)^2/\lvert S\rvert}$ | Mean, P99, maximum |
| Maximum probability error | $\max_{i\in S}\lvert q_i-p_i\rvert$ | Mean, P99, maximum and token |
| Probability-mass coverage | Own top-100 and union mass for each side | Mean, P1, minimum |
| Secondary top-100 overlap | $\lvert T_A\cap T_B\rvert/100$ | Mean |

Weight positions equally and use FP64 offline arithmetic. Compute RMSE per position before aggregation. Retain top-1 mismatch examples with both token probabilities and each side's first/second margin.

Use stable complement accumulation and a log-space mixture for Jensen–Shannon divergence in nats. Zero probability contributes zero. Tail merging compares the union plus one complement category.
