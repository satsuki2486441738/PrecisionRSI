# Inference

## Collect scores

Replay identical token IDs, position IDs, attention masks and prediction positions using the same checkpoint/tokenizer/config and numerical policy. For cache-off full forward, use eval/no-grad. At prediction position $t-1$, collect the distribution predicting token $x_t$.

Compute full-vocabulary natural-log softmax in FP32 before temperature or sampling processors, retaining model-defined scaling/softcapping. Save full logprob rows or queryable arrays, sample/position coordinates and vocabulary mapping.

Feed the same predetermined continuation while each side evolves its native cache. Record prefill/decode stages, batching, cache settings and context length with the scores.

Collect requested label logprobs even when labels lie outside top-100. Use the offline [tools](../methods/tools.md) for distribution and target-score comparison. Group results by prefill/first decode/later decode and requested context-length bins. For generated continuations, pair scores while token histories match.

## Framework integration

Record applicable attention backend, TP, KV dtype, graph/cache/chunking and batching settings. Map reordered requests/chunks to absolute prediction positions: prompt token $x_t$ uses position $t-1$; the final prefill row supplies the first output distribution. Account for BOS/templates and initial prompt entries.

Reconstruct vocabulary shards, exclude padding and export the score rows needed for union queries. Preserve each engine's native cache while replaying a predetermined continuation.

Select the native entrypoint and inspect the installed version's implementation:

| Framework / entrypoint | Capture mapping |
|---|---|
| PyTorch / [Transformers](https://huggingface.co/docs/transformers/main_classes/text_generation) | Native model forward and cache state; distinguish raw logits from processed generation scores |
| [vLLM](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/) | `prompt_logprobs`, output `logprobs`, raw-score mode and full-vocabulary limits; resolve initial `None` entries, chunk offsets and platform plugins |
| [SGLang](https://docs.sglang.io/docs/basic_usage/sampling_params) | `return_logprob`, `logprob_start_len`, top/selected-ID queries and score transforms; preserve radix-cache and extend/decode behavior |
| [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/features/additional-outputs.html) | Installed execution backend, context/generation logits, vocabulary layout and effective build/precision settings |
| [LMDeploy](https://lmdeploy.readthedocs.io/en/latest/) | Selected PyTorch/TurboMind backend, native score outputs and request/token/cache mappings |
| [MindIE-LLM](https://github.com/Ascend/MindIE-LLM) | Service/model backend, available score access and scheduled request/prefill/decode mappings |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF/checkpoint identity, quantization, tokenizer conversion and native context/logit-row mapping |

Resolve score availability at the native execution layer and export the observations required by the selected metrics. Use engine replay or a collector for fixed-history decode.

### Other/custom

Follow [native adaptation](../methods/collection.md#adapt-an-unfamiliar-implementation). Locate the model or engine's native scoring path, resolve score transforms and map outputs to token/vocabulary coordinates. For cached execution, replay the fixed continuation through its native cache and identify prefill/decode boundaries.

## Metrics

| Observations | Report |
|---|---|
| Output distributions | Tail-merged Jensen–Shannon divergence, top-1 agreement, probability RMSE and maximum probability error |
| Requested target scores | Each side's NLL and absolute NLL difference; logprob MAE, P99 and maximum |
| Coverage | Paired positions/targets, each side's top-100 and union mass; secondary top-100 overlap |

Use the [distribution definitions](../methods/metrics.md#distributions) and [target-score definitions](../methods/metrics.md#target-token-scores). Keep the prefill/decode and context-length breakdowns requested by the task.
