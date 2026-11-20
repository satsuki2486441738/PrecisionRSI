# Ascend upstream regression rerun and new cases

On 2026-11-20, NumericBridge ran the [current workflow](../../skills/numericbridge/references/workflow.md) at `main` revision `0bbb7769be0cd9f4c5ec1639f00abf160f53a254`. This run repeats the CPU fixtures in the [earlier report](../ascend-upstream-workflow-20261120/README.md) and adds two upstream changes dated November 19. The live community index returned `no-records` for the new fixture configurations.

## Rerun

The existing workflow completed 18 stages and passed 266 declared checks. All six pinned CPU fixtures reproduced their baseline differences. All six merged or candidate configurations matched their references exactly, and three CPU smoke controls passed. The [fresh measurements](verification.json) include capture timestamps, source identities and coverage.

Four existing native Ascend cases remain blocked at environment preparation. The local runtime lacks `torch_npu`, and neither configured device connection completed preflight.

## Newly verified cases

| Case | What the fixture observes | Baseline unequal | Candidate unequal |
| --- | --- | ---: | ---: |
| [vLLM-Ascend #22270](https://github.com/vllm-project/vllm-ascend/pull/22270), merged | Mooncake sliding-window prefill/decode handoff counters and KV history. The baseline drops token position 511 for a 1,023-token request. | 7 / 1,553 | 0 / 1,553 |
| [MindSpeed-LLM !5235](https://gitcode.com/Ascend/MindSpeed-LLM/merge_requests/5235), open | The DSA indexer loss overwrites three saved `-1` padding indices. The CPU gather proxy exposes changes in outputs and KV gradients. | 6 / 34 | 0 / 34 |

Both cases passed source readiness, CPU preflight and both comparison stages. The vLLM case executes pinned scheduler, cache allocation and eviction methods using synthetic requests and an in-memory page queue. The MindSpeed-LLM case executes the production indexer loss on CPU and uses a synthetic sparse consumer to assess the saved-index contract. Acceptance is exact in each fixture's declared scope.

The [community draft](submission.json) contains the two new repair records and their measured evidence. It passes the latest submission preview validator. The [live community service](https://satsuki2486441738.github.io/NumericBridge/data/community-service.json) reports `submission: null`, so the draft remains at `local_draft`.

## Reproduce the new cases

From the NumericBridge root, use Python with PyTorch and NumPy. Fetch the pinned source objects before starting each offline workflow.

```sh
base=$(mktemp -d)
runner=reports/ascend-upstream-workflow-rerun-20261120/run.py

git init "$base/mindspeed-llm"
git -c http.version=HTTP/1.1 -C "$base/mindspeed-llm" fetch --depth=1 --no-tags \
  https://gitcode.com/Ascend/MindSpeed-LLM.git \
  4458c30c2deff2534d5373c6002b8a16c8fb0551 259fa5991088c869d78c2e0f601a0fa096648c47
python -B "$runner" run mindspeed-llm-dsa-topk-padding \
  --source "$base/mindspeed-llm" --workspace "$base/dsa-run"

git init "$base/vllm-ascend"
git -C "$base/vllm-ascend" fetch --depth=2 --no-tags \
  https://github.com/vllm-project/vllm-ascend.git 47b32942d6e74b757fcae48cd40c181e53c7e588
git init "$base/vllm-ascend/vllm"
git -C "$base/vllm-ascend/vllm" fetch --depth=1 --no-tags \
  https://github.com/vllm-project/vllm.git ba15685ab7f4f5fbed8630bbb7441f8837e76fbe
python -B "$runner" run vllm-ascend-mooncake-swa-handoff \
  --source "$base/vllm-ascend" --workspace "$base/mooncake-run"
```
