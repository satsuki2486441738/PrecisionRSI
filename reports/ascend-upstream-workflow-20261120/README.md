# Ascend upstream regression workflow results

NumericBridge reran six upstream regressions and three CPU controls on 2026-11-20 using the [workflow and contribution guide](../../skills/numericbridge/references/workflow.md) from [main at `0bbb776`](https://github.com/satsuki2486441738/NumericBridge/commit/0bbb7769be0cd9f4c5ec1639f00abf160f53a254). The fixtures use pinned upstream Git objects, small deterministic inputs and independent references. The baseline and candidate run the same fixture. The source snapshots and fixture definitions are traceable to [the original workload commit `77e18cd`](https://github.com/satsuki2486441738/NumericBridge/tree/77e18cd8a7ce1d32bc8197f337956316d3752639/reports/ascend-upstream-cases-20261119).

## Results

All six CPU source fixtures reproduced the discrepancy at the baseline revision and matched their references exactly at the merged or candidate revision. All three CPU smoke controls matched. Each comparison covered its full declared observation inventory.

| CPU source fixture | Baseline unequal | Candidate unequal | Upstream |
| --- | ---: | ---: | --- |
| SGLang MLA prefix cache compaction | 772 / 3,208 | 0 / 3,208 | [#47924](https://github.com/sgl-project/sglang/pull/47924), merged |
| SGLang GDN recurrent state layout | 480 / 704 | 0 / 704 | [#50245](https://github.com/sgl-project/sglang/pull/50245), open |
| vLLM-Ascend DeepSeek V4 attention sink reload | 144 / 144 | 0 / 144 | [#18642](https://github.com/vllm-project/vllm-ascend/pull/18642), merged |
| MindSpeed sequence-parallel linear bias | 16 / 224 | 0 / 224 | [!4128](https://gitcode.com/Ascend/MindSpeed/merge_requests/4128), open |
| MindSpeed packed MTP first-segment shift | 92 / 115 | 0 / 115 | [!4115](https://gitcode.com/Ascend/MindSpeed/merge_requests/4115), merged |
| MindSpeed-LLM GPT dataset loss mask isolation | 4 / 16 | 0 / 16 | [!5212](https://gitcode.com/Ascend/MindSpeed-LLM/merge_requests/5212), merged |

The comparisons execute the pinned Python paths on macOS arm64 using PyTorch 2.15.0 and NumPy 2.4.2. The GDN kernel boundary, vLLM deferred replay lifecycle and MindSpeed tensor-parallel collectives use CPU fixtures. Acceptance applies to each listed output inventory. The two open changes were tested at their candidate heads.

## Workflow and device readiness

The [live index](https://satsuki2486441738.github.io/NumericBridge/data/search-index.json) returned `no-records` for the six fixture configurations. `tools.workflow` completed 18 of 18 stages with 266 declared assertions passing. It checked pinned source and CPU runtime readiness, ran 12 baseline and candidate captures and three controls, then reconciled the results. The array comparator recorded coverage, nonfinite counts, and acceptance for each fixture.

Four native paths are pending device execution. The local runtime lacks TorchNPU and the configured Ascend connections could not complete preflight. The CPU control outcomes cover reference arithmetic and the comparison path.

| Native path pending | Scope | Upstream |
| --- | --- | --- |
| TorchNPU Inductor `expm1` | Compiled FP32 values near zero | [!49365](https://gitcode.com/Ascend/pytorch/merge_requests/49365) |
| TorchNPU Triton int64 division | Compiled floor/trunc division and fusion | [!49540](https://gitcode.com/Ascend/pytorch/merge_requests/49540) |
| op-plugin host-scalar multiplication | BF16 and FP16 bit patterns | [!6062](https://gitcode.com/Ascend/op-plugin/merge_requests/6062) |
| vLLM-Ascend QSFA TND output | Valid-row finiteness and zero padding after capacity shrink | [#20578](https://github.com/vllm-project/vllm-ascend/pull/20578) |

## Community review

The [submission draft](submission.json) contains the six repair records, their lifecycle relations and measured evidence. NumericBridge validated the draft against `0bbb776` and compiled a candidate search index in its isolated review checkout. The [live service descriptor](https://satsuki2486441738.github.io/NumericBridge/data/community-service.json) reported `submission: null`. The contribution is a `local_draft` awaiting an available inbox.

From the NumericBridge repository root, reproduce the local preview against the current `origin/main` using a fresh output path.

```sh
python -B -m community.records.submission \
  --input reports/ascend-upstream-workflow-20261120/submission.json \
  --output /tmp/ascend-upstream-community-preview.json
```
