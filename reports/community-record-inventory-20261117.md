# Community record inventory — 2026-11-17

This is a reviewed association inventory, not a new model experiment or numerical
acceptance report. Original evaluations, repairs, measurements and published source
evidence remain unchanged. The additive [relation catalog](../community/site/data/relations.json)
is the canonical case/issue/solution/verification mapping; metrics remain in their
original records and evidence. Use the [record interface](../community/records/README.md)
for queries and source resolution. A separate `index.html#/cases` lifecycle preview
now presents these associations alongside the existing website views.

## Coverage

| Retained content | Indexed coverage | Interpretation |
|---|---:|---|
| Evaluations | 130 / 130 | 104 precision-evaluation cases + 26 Skills/Agent workflow-benchmark cases |
| Repair histories | 20 / 20 | 13 root cases; explicit children remain follow-up evidence |
| Measurement views | 7783 / 7783 source-indexed | 127 exact catalog-parent associations; 7656 source-only views |
| Cases | 143 | Retrieval/grouping boundaries, not independent experiments |
| Issues | 14 | 11 historical confirmed defects, 2 numerical discrepancies, 1 unconfirmed hypothesis |
| Solutions | 27 | 18 runtime-tested in scope; 9 source-inspected upgrade routes |
| Verification associations | 21 | Retained original workloads; no new native execution |

All 130 evaluation and 20 repair acceptance fields remain null. Legacy completion
or `validated` labels are preserved, not translated into numerical acceptance.
Recorded task labels do not certify full RL, long training, complete model coverage
or hardware causality. Source-inspected upgrades have no runtime `verifies` link;
fixed-version boundaries, earliest releases and present-day upstream status remain
unknown unless explicitly supported by the retained source.

## Repair case map

Each row groups only its declared records. Solution counts include unselected or
source-only alternatives. Verification counts are scope-specific associations.

| Case ID | Legacy repairs | Issues | Solutions | Verifications |
|---|---:|---:|---:|---:|
| `case-mindspeed-zero-centered-rmsnorm` | 6 | 2 | 2 | 3 |
| `case-qwen35-cross-platform-rmsnorm-policy` | 1 | 1 | 4 | 4 |
| `case-trl-ppo-whitening-tail` | 1 | 1 | 2 | 1 |
| `case-radam-unrectified-zero-gradient` | 1 | 1 | 1 | 1 |
| `case-dora-conv1d-unmerge` | 1 | 1 | 2 | 1 |
| `case-asgd-foreach-schedule` | 2 | 1 | 2 | 2 |
| `case-trainer-label-smoothing` | 1 | 1 | 2 | 1 |
| `case-ppo-response-mask` | 1 | 1 | 2 | 1 |
| `case-ipo-completion-average` | 1 | 1 | 2 | 1 |
| `case-lora-svd-double-scaling` | 2 | 1 | 2 | 2 |
| `case-qwen2-disabled-window` | 1 | 1 | 2 | 1 |
| `case-qwen35-weighted-lora-bias` | 1 | 1 | 3 | 2 |
| `case-qwen35-token-mean` | 1 | 1 | 1 | 1 |

Important bounded interpretations:

- ASGD executes historical v2.3.0 optimizer source on newer actual runtimes. The
  9B LoRA follow-up is additional verification, not a defect in the newer runtime.
- The SVD Agent forwardtest reuses the original known issue/correction; it does not
  establish another discovery or independent solution.
- The weighted-bias PEFT case retains the rejected bias-only backport, the final
  local FP32 accumulator correction and the source-only upstream route separately.
- Qwen3.5 RMSNorm has four numerical-policy candidates. The selected backward-only
  policy remains partial; operator cost is not end-to-end training overhead.
- The [token-mean scope clarification](../community/site/evidence/token-mean-scope-clarification-20261117.md)
  preserves the historical stronger prose and explains the unconfirmed issue:
  manual `training_step`, selected gradients and common accumulation scaling do
  not establish native full-loop semantics, absolute scaling or optimizer updates.
- Tiny CPU components, full pretrained models and partial update captures retain
  distinct scopes. No library defect is inferred from a numerical difference alone.

## Evaluation case inventory

Each original evaluation has a `case-evaluation-<evaluation-id>` retrieval wrapper.
No defect, solution or causal repair relationship is inferred by that wrapper.
Recorded statuses and task labels below retain their original meaning; all
acceptances are null. `precision` abbreviates precision-evaluation and `workflow`
abbreviates workflow-benchmark.

| Evaluation ID | Category | Recorded task | Recorded status |
|---|---|---|---|
| `trl-sfttrainer-checkpoint-20261021` | precision | sft | completed |
| `trl-sfttrainer-lora-20261021` | precision | sft | completed |
| `torch-bf16-fp16-20261021` | precision | inference | completed |
| `llamacpp-cann-qwen3-0-6b-20261020` | precision | inference | completed |
| `llamacpp-cann-qwen3-0-6b-base-20261020` | precision | inference | completed |
| `lmdeploy-ascend-qwen3-0-6b-20261020` | precision | inference | completed |
| `lmdeploy-ascend-qwen3-0-6b-base-20261020` | precision | inference | completed |
| `tensorrt-llm-seeded-llama-20261020` | precision | inference | completed |
| `tensorrt-llm-qwen25-05b-20261020` | precision | inference | completed |
| `mindie-qwen3-06b-20261020` | precision | inference | completed |
| `llamacpp-luna-smollm2-135m-20261019` | precision | inference | completed |
| `llamacpp-local-smollm2-135m-instruct-20261019` | precision | inference | completed |
| `lmdeploy-local-smollm2-135m-instruct-20261019` | precision | inference | completed |
| `llamacpp-local-qwen2-5-0-5b-instruct-20261019` | precision | inference | completed |
| `lmdeploy-local-qwen2-5-0-5b-instruct-20261019` | precision | inference | completed |
| `trl-tiny-sft-npu-20261020` | precision | sft | completed |
| `trl-pretrained-sft-npu-20261020` | precision | sft | completed |
| `factory-tiny-sft-npu-20261020` | precision | sft | completed |
| `factory-pretrained-sft-npu-20261020` | precision | sft | completed |
| `swift-tiny-sft-npu-20261020` | precision | sft | completed |
| `swift-pretrained-sft-npu-20261020` | precision | sft | completed |
| `axolotl-tiny-sft-npu-20261020` | precision | sft | completed |
| `axolotl-pretrained-sft-npu-20261020` | precision | sft | completed |
| `megatron-bridge-tiny-sft-npu-20261020` | precision | sft | completed |
| `megatron-bridge-pretrained-sft-npu-20261020` | precision | sft | completed |
| `mindspeed-llm-mini-pretraining-npu-20261020` | precision | sft | completed |
| `trl-tiny-qwen2-sft-local-20261019` | precision | sft | completed |
| `trl-smollm2-sft-local-20261019` | precision | sft | completed |
| `llamafactory-tiny-qwen2-sft-local-20261019` | precision | sft | completed |
| `llamafactory-smollm2-sft-local-20261019` | precision | sft | completed |
| `swift-tiny-qwen2-sft-local-20261019` | precision | sft | completed |
| `swift-smollm2-sft-local-20261019` | precision | sft | completed |
| `openrlhf-community-grpo-qwen3-06b-20261020` | precision | rl | completed |
| `slime-grpo-qwen3-06b-20261020` | precision | rl | completed |
| `nemo-rl-tiny-qwen2-worker-20261020` | precision | rl | completed |
| `nemo-rl-qwen3-06b-official-cli-20261020` | precision | rl | completed |
| `ms-swift-grpo-qwen3-06b-20261020` | precision | rl | completed |
| `areal-grpo-qwen3-06b-20261020` | precision | rl | completed |
| `roll-grpo-qwen3-06b-20261020` | precision | rl | completed |
| `mindspeed-rl-grpo-tiny-qwen3-20261020` | precision | rl | completed |
| `openrlhf-rl-local-tiny-qwen2-20261019` | precision | rl | completed |
| `openrlhf-rl-local-smollm2-135m-20261019` | precision | rl | completed |
| `ms-swift-rl-local-smollm2-135m-20261019` | precision | rl | completed |
| `axolotl-qwen3-06b-cuda-native-20261021t182602z` | precision | sft | completed |
| `axolotl-tiny-qwen2-cuda-native-20261021t182602z` | precision | sft | completed |
| `factory-qwen3-06b-cuda-native-20261021t181049z` | precision | sft | completed |
| `factory-tiny-qwen2-cuda-native-20261021t181049z` | precision | sft | completed |
| `swift-qwen3-06b-cuda-native-20261021t182317z` | precision | sft | completed |
| `swift-tiny-qwen2-cuda-native-20261021t182317z` | precision | sft | completed |
| `trl-qwen3-06b-cuda-native-20261021t175700z` | precision | sft | completed |
| `trl-tiny-qwen2-cuda-native-20261021t175700z` | precision | sft | completed |
| `lmdeploy-cuda-smollm2-135m-instruct-20261021` | precision | inference | completed |
| `llama-cuda-smollm2-135m-instruct-20261021` | precision | inference | completed |
| `lmdeploy-cuda-qwen25-05b-instruct-20261021` | precision | inference | completed |
| `llama-cuda-qwen25-05b-instruct-20261021` | precision | inference | completed |
| `swift-tiny-fp32-cuda-20261021` | precision | rl | completed |
| `swift-pretrained-bf16-cuda-20261021` | precision | rl | completed |
| `slime-tiny-20261021t194857z` | precision | rl | completed |
| `slime-pretrained-20261021t195747z` | precision | rl | completed |
| `nemo-tiny-untied-fp32-h100-20261022` | precision | rl | completed |
| `nemo-pretrained-untied-fp32-h100-20261022` | precision | rl | completed |
| `trl-qwen3-06b-h100-ddp-native-20261022` | precision | sft | completed |
| `trl-tiny-qwen2-h100-ddp-native-20261022` | precision | sft | completed |
| `areal-tiny-cuda-h100-20261022` | precision | rl | completed |
| `areal-pretrained-cuda-h100-20261022` | precision | rl | completed |
| `openrlhf-tiny-cuda-h100-20261022` | precision | rl | completed |
| `openrlhf-pretrained-cuda-h100-20261022` | precision | rl | completed |
| `roll-tiny-cuda-h100-20261022` | precision | rl | completed |
| `roll-pretrained-cuda-h100-20261022` | precision | rl | completed |
| `megatron-bridge-qwen3-06b-h100-native-20261022` | precision | sft | completed |
| `megatron-bridge-tiny-qwen3-h100-native-20261022` | precision | sft | completed |
| `skills-ab-cache-20261024` | workflow | inference | completed |
| `skills-ab-sft-20261024` | workflow | sft | completed |
| `skills-ab-rl-20261024` | workflow | rl | completed |
| `torch-native-multitask-inference-20261024` | precision | inference | completed |
| `torch-native-multitask-sft-20261024` | precision | sft | completed |
| `torch-native-multitask-rl-20261024` | precision | rl | completed |
| `skills-demand-cache-20261024` | workflow | inference | completed |
| `skills-demand-sft-20261024` | workflow | sft | completed |
| `skills-demand-rl-20261024` | workflow | rl | completed |
| `skills-wait-sft-20261024` | workflow | sft | completed |
| `skills-demand-native-prefill-20261024` | workflow | inference | completed |
| `skills-interface-native-prefill-20261024` | workflow | inference | completed |
| `skills-tool-policy-microtest-20261024` | workflow | inference | completed |
| `skills-integrity-reuse-microtest-20261024` | workflow | inference | completed |
| `skills-original-native-retest-20261024` | workflow | inference | completed |
| `skills-single-checkpoint-retest-20261024` | workflow | sft | completed |
| `skills-single-library-ab-20261024` | workflow | inference | completed |
| `skills-single-library-clear-20261024` | workflow | inference | completed |
| `skills-continuous-reuse-before-20261024` | workflow | sft | completed |
| `skills-continuous-reuse-bullet-20261024` | workflow | sft | completed |
| `skills-continuous-reuse-table-20261024` | workflow | sft | completed |
| `skills-conditional-reuse-20261024` | workflow | sft | completed |
| `skills-retained-results-20261024` | workflow | sft | completed |
| `skills-novel-ddp-a-20261024` | workflow | sft | completed |
| `skills-novel-shards-a-20261024` | workflow | inference | completed |
| `skills-novel-shards-a1-20261024` | workflow | inference | completed |
| `skills-novel-shards-a2-20261024` | workflow | inference | completed |
| `skills-native-sglangtf-initial-20261024` | workflow | inference | completed |
| `skills-native-sglangtf-extra-20261024` | workflow | inference | completed |
| `autonomous-cross-platform-inference-20261025` | precision | inference | completed |
| `autonomous-cross-platform-sft-20261025` | precision | sft | completed |
| `cross-platform-sft-causal-continuation-20261025` | precision | sft | completed |
| `gemma4-h100-rtx5090-20261105` | precision | inference | partial |
| `gemma4-ascend-tp1-repeat-20261106` | precision | inference | partial |
| `gemma4-h100-ascend-fixed-20261106` | precision | inference | partial |
| `gemma4-h100-ascend-cacheoff-20261106` | precision | inference | partial |
| `gemma4-h100-ascend-partitions-20261107` | precision | inference | partial |
| `gemma4-h100-ascend-batch-padding-20261107` | precision | inference | partial |
| `gemma4-h100-observer-transparency-20261107` | precision | inference | partial |
| `gemma4-h100-ascend-observer-transparency-20261107` | precision | inference | partial |
| `qwen38-native-inference-20261106` | precision | inference | completed |
| `qwen38-native-lora-frameworks-20261106` | precision | sft | completed |
| `qwen38-native-rl-components-20261106` | precision | rl | completed |
| `qwen38-native-serving-admissions-20261106` | precision | inference | partial |
| `deepseek41-original-vs-adapted-core-20261106` | precision | inference | partial |
| `deepseek41-upstream-compressor-key-local-vjp-20261108` | precision | inference | partial |
| `deepseek41-native-query-tail-20261108` | precision | inference | partial |
| `deepseek41-adapted-core275-20261109` | precision | inference | partial |
| `deepseek41-full40-attention-20261109` | precision | inference | partial |
| `deepseek41-paired-output-head95-20261109` | precision | sft | partial |
| `deepseek41-http128-20261109` | precision | inference | partial |
| `deepseek41-native-producer-key-cache-20261109` | precision | inference | partial |
| `deepseek41-native-query-key-selection-20261109` | precision | inference | partial |
| `gemma-3-27b-it-cuda-precision-20261110` | precision | inference | completed |
| `gpt-oss-120b-cuda-precision-20261110` | precision | inference | completed |
| `nemotron-bolt-cuda-precision-20261110` | precision | inference | completed |
| `gemma-3-27b-it-ascend-precision-20261110` | precision | inference | completed |
| `gpt-oss-120b-ascend-precision-20261110` | precision | inference | completed |
| `nemotron-bolt-ascend-precision-20261110` | precision | inference | completed |

## Evidence and unresolved associations

The [immutable association ledger](../community/site/evidence/community-record-associations-20261117.json)
contains exact source pointers and canonical hashes for the 127 measurement-parent
matches. The full raw source, original parent and complete comparison hashes are
retained; task/model/side identity must also match. Views from standalone frozen
numerical sources remain indexed under their original source IDs. Model or
filename similarity alone is not an association rule.

Six evaluations have only historical external evidence links. Their retained
catalog records are hash-bound in the ledger, but the referenced archives were
not fetched or replayed during this organization. They remain discoverable with
that explicit limit. This is not a historical model rerun or proof of external
archive availability.

Case request fields summarize historical scope, not a retroactive user confirmation.
Unrecorded frameworks/tasks remain null; exact configuration filters read actual
member sides/environments rather than guessed library roles. The existing board
still reads its original catalogs; a separate Cases preview reads the explicit
associations, also accessible through the Python package and CLI. Historical
unknowns require new scoped evidence, rather
than rewriting old records.

## Reproduce the inventory

```sh
python -B -m community.records validate
python -B -m community.records inventory
python -B -m community.records cases --category repair-investigation
python -B -m community.records cases --hardware 'Ascend 910C' --library torch_npu --task rl
python -B -m community.records case --id case-qwen35-weighted-lora-bias
python -B -m scripts validate-immutability --base origin/main
```

Record queries are read-only. Hardware and library/version must match one actual
member configuration together. Reading a candidate upgrade is not validation of
that upgraded environment. Original full arrays, execution logs, archived packages
and local developer regressions remain outside the public repository.
