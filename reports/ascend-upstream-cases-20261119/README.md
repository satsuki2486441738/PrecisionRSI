# Ascend upstream precision regressions

Collected on 2026-11-19 from GitCode merge requests and GitHub pull requests. Screening covered native Ascend/PyTorch, op-plugin and CANN repositories and Ascend integrations in vLLM, SGLang and training frameworks. The fixture set retains changes with a specific numerical symptom, a small deterministic input or seed, a reference or invariant, and pinned upstream revisions. The selected changes opened or merged from 2026-10-27 through 2026-11-18.

| Case ID | Numerical check | Upstream | Mode |
| --- | --- | --- | --- |
| `torchnpu-inductor-expm1` | Compiled FP32 `expm1` near zero against an FP64 reference rounded to FP32 | [Ascend/pytorch !49365](https://gitcode.com/Ascend/pytorch/merge_requests/49365), merged Nov 18 | `native-npu`, `cpu-smoke` |
| `torchnpu-triton-int64-div` | Exact floor/trunc division near `±2**40`, host/device scalars and a fused expression | [Ascend/pytorch !49540](https://gitcode.com/Ascend/pytorch/merge_requests/49540), merged Nov 17 | `native-npu`, `cpu-smoke` |
| `sglang-gdn-state-layout` | GDN prefill attention, final recurrent state and chunk snapshots | [SGLang #50245](https://github.com/sgl-project/sglang/pull/50245), opened Nov 13 | `source-cpu` |
| `sglang-mla-prefix-compaction` | Valid cache tokens and packed row counts for ND/NZ with mixed and page-aligned lengths | [SGLang #47924](https://github.com/sgl-project/sglang/pull/47924), merged Nov 4 | `source-cpu` |
| `vllm-ascend-deepseek-v4-sink-reload` | Two attention-sink updates through target, MTP and DSpark loaders with TP rank-local and DSA CP layouts | [vLLM-Ascend #18642](https://github.com/vllm-project/vllm-ascend/pull/18642), merged Nov 3 | `source-cpu` |
| `vllm-ascend-qsfa-tnd-tail` | Finiteness of 36 valid query rows and exact zero across 28 padding rows after a 128→64 capacity shrink | [vLLM-Ascend #20578](https://github.com/vllm-project/vllm-ascend/pull/20578), merged Nov 2 | `native-npu` |
| `op-plugin-mul-host-scalar` | BF16 and FP16 bit patterns for FP32 host-scalar multiplication, including `out=` | [Ascend/op-plugin !6062](https://gitcode.com/Ascend/op-plugin/merge_requests/6062), merged Oct 27 | `native-npu`, `cpu-smoke` |

The GDN case targets an open PR's candidate head. The QSFA padding check can be allocator-dependent on the baseline, so a baseline match is inconclusive. The `source-cpu` cases execute affected pinned Python code with synthetic tensors and small substitutes at runtime boundaries. The `cpu-smoke` controls exercise the reference and artifact path using eager CPU operations. Native runs exercise the installed Ascend stack.

On 2026-11-19, all three source cases reproduced the discrepancy at the baseline revision and matched their references at the candidate or merged revision. All three CPU smoke controls matched. The four native NPU cases await device execution. The [measured results](verification.json) include scope, acceptance and the environment for each CPU run.

## List and run cases

Use Python 3.10 or later with NumPy and PyTorch. Run the examples from the NumericBridge repository root.

```sh
runner=reports/ascend-upstream-cases-20261119/run_case.py
python -B "$runner" list
python -B "$runner" run op-plugin-mul-host-scalar \
  --mode cpu-smoke --variant after --expect match
```

Each run writes `reference.npz`, `observed.npz`, `expected.json`, `report.json` and `capture.json` under a fresh UTC-named directory. The [NumericBridge array comparator](../../skills/precision-compare/scripts/array_compare.py) writes coverage, nonfinite counts, error metrics and mismatching coordinates. `capture.json` records the case, revision, environment and acceptance. Acceptance requires equal finite values, shapes and dtypes. `expm1` allows relative tolerance `1e-6` and zero absolute tolerance. With `--expect`, an unexpected outcome exits with code 1. With no expectation, either complete comparison exits with code 0. Set `--output-root` to choose the local artifact location.

## Run the pinned CPU source

Fetch once into scratch repositories or use existing local Git repositories with these objects. The runner selects source by revision without switching the working tree. No model checkpoint is required.

```sh
git init /tmp/nb-sglang-source
git -c http.version=HTTP/1.1 -C /tmp/nb-sglang-source fetch --depth=1 --no-tags https://github.com/sgl-project/sglang.git \
  2120cf1190d32135358ea0fa73a87c4ca232803f 643909b5a898665aa7640c85b48963598f41815a \
  c2578f28cfaece96167765ed93cab9a484f6fe95 8e5404fa8b8a1e12f7029c7e7c17dbfed7f89475

git init /tmp/nb-vllm-ascend-source
git -C /tmp/nb-vllm-ascend-source fetch --depth=1 --no-tags https://github.com/vllm-project/vllm-ascend.git \
  cb629ea1925a280d57b41b48416a85097f6cdb6b db9ad13183f6836f7f83304590e1ba4035b888e5

case_id=sglang-mla-prefix-compaction
source_checkout=/tmp/nb-sglang-source
python -B "$runner" run "$case_id" --source "$source_checkout" \
  --mode source-cpu --variant before --expect mismatch
python -B "$runner" run "$case_id" --source "$source_checkout" \
  --mode source-cpu --variant after --expect match
```

Repeat with `sglang-gdn-state-layout` in the same checkout and `vllm-ascend-deepseek-v4-sink-reload` using `/tmp/nb-vllm-ascend-source`. MLA uses mock buffers and forward metadata for one DCP rank. GDN supplies a CPU implementation of the external chunk kernel to the real Python wrapper. The vLLM-Ascend reload case executes the three loader functions with a sink-only deferred replay fixture and a cold-load control.

## Run on Ascend

Use `list` for full revisions. Build or install each revision in a compatible environment. Use `--runtime-label` to record the installed build revision, NPU SKU and CANN version. Run the same case ID with `--variant before` and `--variant after` in their respective environments. `--expect mismatch` and `--expect match` turn the suspected regression and the expected fix into explicit checks.

- For the TorchNPU cases, install the matching PyTorch, TorchNPU, Triton and CANN stack. `expm1` uses the default Triton Inductor backend, while int64 division uses experimental Ascend Triton Inductor. [TorchNPU source build](https://gitcode.com/Ascend/pytorch/blob/master/docs/zh/installation_guide/building_from_source.md) · [expm1 MR](https://gitcode.com/Ascend/pytorch/merge_requests/49365) · [int64 division MR](https://gitcode.com/Ascend/pytorch/merge_requests/49540)
- For host-scalar multiplication, build TorchNPU twice with `third_party/op-plugin` at each selected revision and use the same TorchNPU base, PyTorch, CANN and device. The upstream report used Ascend 910B2, CANN 9.2.0 and PyTorch 2.13. The arrays retain 16-bit encodings. The CPU control multiplies in FP32 before casting, consistent with the reference rounding rule. [op-plugin build](https://gitcode.com/Ascend/op-plugin/blob/cb832887c0dfefff9109ee04b6a98765157a980a/README.md) · [op-plugin MR](https://gitcode.com/Ascend/op-plugin/merge_requests/6062)
- For QSFA, use a compatible vLLM, PyTorch, TorchNPU and CANN stack with one Ascend device. Build the custom operator at each selected revision with `pip install -e .` and pass that checkout via `--source`. [Native regression](https://github.com/vllm-project/vllm-ascend/blob/9b6770e239b1f919d451c63a290a7ef09e8c9dc1/tests/e2e/nightly/single_node/ops/singlecard_ops/test_kv_quant_sparse_flash_attention_vllm.py)

```sh
python -B "$runner" run torchnpu-triton-int64-div --mode native-npu --variant after \
  --device npu:0 --runtime-label "$ASCEND_BUILD_DESCRIPTION" --expect match
```
