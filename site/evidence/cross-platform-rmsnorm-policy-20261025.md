# Qwen3.5 RMSNorm: partial cross-platform numerical-policy repair

Torch2.9, Transformers5.20.2, PEFT0.20.0; Ascend910C and the confirmed H200 cluster. This is a numerical-policy candidate, not a confirmed upstream library defect.

The selected candidate preserves the source-order native RMSNorm forward and uses fixed FP32 pairwise reductions plus an independently computed inverse RMS in the analytic backward. All81 standard Qwen3_5RMSNorm modules are covered; gated norms, QKV projections and native fused attention stay unchanged. The attached Python module exposes `install(model, backward_only=True)` after model/LoRA loading and before training. Large model resources and full execution artifacts remain in the private case.

The local common-input test reproduces native forward and matches input gradients across the two peers. The mathematical backward was checked against an independent CPU FP64 formula on actual saved model inputs/cotangents. The final integer division has a registered NPU kernel, matches the CPU integer result and preserves the earlier stable-gradient output. The first prototype used an unsupported right shift and produced CPU fallback; that prototype is retained separately.

The original complete32-layer SFT uses the same checkpoint, initial adapters,138 input positions and123 supervised targets. All80 trainable tensors and671,744 coordinates are covered. Both peers retain exactly their own original nonpadding logits and original-exporter target scores.

| Original-task comparison | Native baseline | Selected backward policy |
|---|---:|---:|
| Target NLL absolute gap | 0.001893141202 | unchanged |
| Gradient error L2 | 0.006179175213 | 0.005948786426 |
| Gradient cosine | 0.999241361935 | 0.999276003076 |
| Stored-update error L2 | 0.012798216513 | 0.012690478891 |
| Stored-update cosine | 0.976142557324 | 0.976543318450 |
| Opposite-sign gradient coordinates | 4283 | 4200 |
| Maximum stored-update difference | 0.000199988339 | 0.000199988768 |

The original layer7 v_proj LoRA B[302,1] gradient remains opposite-sign. This is a bounded aggregate improvement and local reproducibility repair, not complete SFT equivalence. First-step stored updates independently agree with the AdamW formula within2.45e-11 /3.20e-11.

Actual RMSNorm forward+backward operator median cost is2.28x native on GPU and2.64x on NPU, measured at shape(1,138,4096), with two warmups and five synchronized samples. These ratios do not measure end-to-end training overhead. Forward-changing all-module, QKV-combined and single-layer alternatives are retained in the companion JSON with their metric tradeoffs.

Private case: `projects/precision-repair/cases/cross-sft-rmsnorm-repair-20261025T074434Z`. The companion JSON records source snapshots/hashes, identities, configurations, before/after metrics, coverage, native-forward preservation and independent checks.
