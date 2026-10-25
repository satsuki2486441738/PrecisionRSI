# Qwen3.5-9B-Base: cross-platform SFT causal continuation

Ascend 910C and NVIDIA H200, pinned Torch 2.9 / Transformers 5.20.2 / PEFT 0.20.0. Complete 32-layer model, 123 supervised tokens, all 80 LoRA tensors (671,744 elements), one AdamW step.

No library defect or overall repair was established. Four isolated precision policies were evaluated against the original workload. QKV FP32 accumulation improves gradient/update agreement but worsens the average NLL gap. Explicit FP32 attention reduces the average NLL gap but increases per-token MAE/P99; it replaces the fused backend. Combining both does not resolve the tradeoff. Layer7 output-projection backward FP32 does not change the originally implicated gradient coordinate.

Common-input/cotangent layer7 attention replay reproduces the same target gradient scalar across devices. Holding a saved GPU cotangent fixed at decoder layer16 leaves 76.98% of the original target-gradient gap; holding it at layer8 leaves 5.24% and removes the sign difference. GPU self-replay reproduces the original adapter gradients exactly. This implicates reverse propagation through layers8–15 at each side's actual forward state; differing states and local arithmetic remain unseparated. These interventions are diagnostics, not repairs.

Sixteen native jobs were submitted. The layer16 pair saved the complete native model computation but failed in CPU postprocessing due to a parameter-name lookup; CPU-only recovery verified the saved artifacts. The layer8 pair completed normally. Original failed receipts remain preserved. The error belongs to the case's diagnostic script, not the library.

The companion JSON preserves measured metrics, coverage, diagnostic preservation checks, resource totals and executed source snapshots. Detailed runs stay in the private case `projects/precision-repair/cases/cross-sft-repair-continuation-20261025T052638Z`. Latest/main runtime and full-parameter/multistep/distributed training were not tested.
