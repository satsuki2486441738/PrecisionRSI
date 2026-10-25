# Autonomous cross-platform investigation — Qwen3.5-9B-Base

Two fresh workers independently investigated full-model inference and SFT on Ascend 910C and the user-confirmed H200 platform (runtime name H20Z). The parent completed allocated-GPU captures after resolving resource identity checks. The investigation found measurable cross-platform differences, but no established library defect or library repair.

## Inference

Transformers 5.20.2, eager attention, matching fixed histories and checkpoint. Each precision compares 10 positions over the complete 248,320-token vocabulary. CUDA/CANN native implementations remain in use.

| Scope | BF16 maximum logit difference | FP32 maximum logit difference |
|---|---:|---:|
| 65-token sample, four positions | 0.125 | 3.552436828613281e-5 |
| 257/145-token batch, six positions | 0.25 | 3.5762786865234375e-5 |

All 20 paired position/precision observations have equal top-1. FP32 is loaded directly from the original checkpoint, which contains both BF16 and FP32 tensors. These measurements do not set an application tolerance.

## Native accelerated SFT

Complete 32-layer language backbone, BF16, rank-2 LoRA, exactly equal initial adapters, one AdamW step. CUDA Flash SDPA and torch_npu.npu_fusion_attention are integrated through the same case-owned attention adapter; other model operators retain their native paths.

- Targets: 123/123; gradient/update tensors: 80/80; coordinates: 671,744.
- GPU/NPU NLL: 0.5441815106418652 / 0.546074651843421; absolute difference 0.0018931412015558147.
- Target logprob MAE: 0.004231977514158149; P99: 0.081935454889997.
- Gradient cosine: 0.999241361934603; update cosine: 0.976142557323779.
- Independent first-step AdamW maximum formula error: GPU 2.1827872842550278e-11; NPU 2.9103830456733704e-11. At the maximum cross-device update difference, the supplied gradients have opposite signs and both updates agree with the formula.
- The sampled real first-attention-layer/head-0 forward and gradient checks against FP64 give similar local errors on both sides. They do not identify a whole-model defect.

## Execution and retained evidence

The existing device resolver binds the scheduler-allocated NVML UUID through PCI order and verifies the selected CUDA-driver PCI with one visible CUDA device. Torch and NVML UUID strings are retained separately. Earlier unverified GPU results and failed string-equality guards are excluded. Valid NPU captures were reused; the parent needed two GPU jobs to complete the missing scope. No installed library or shared skill was patched.

The JSON contains measured metrics, coverage, actual configurations, binding evidence, fetch receipts and source snapshots. Full weights and large captures remain in the private native cases. Main was checked at source level only; training covers one LoRA update, not full-parameter or long training. No repair-history entry is added for this investigation.
