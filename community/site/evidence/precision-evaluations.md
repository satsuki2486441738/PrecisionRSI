# Numerical evaluation evidence

These completed evaluations use Qwen3.5-9B-Base revision `68c46c4b3498877f3ef123c856ecfde50c39f404`. The website publishes measured values, coverage and configuration. Completion describes collection/comparison execution; numerical acceptance remains a separate field on each record.

## Same-H200 BF16 and FP16 inference

Two dtypes ran sequentially within one physical allocation. Full forward contains 70 observations; fixed-history cached inference contains 8. Those cached histories also occur in full-forward mode. Each observation retains target coordinates; all 78 target/distribution observations are paired. Target logprobs use full-vocabulary normalization. Top-1 agrees at all 78 observations. Offline arithmetic was independently recomputed and 14 replay checks matched.

- ModelScope dataset: `XinyuanXie/precision-skill-test`.
- Prefix: `torch-dtype-validation/compare-inference-20261021T100207Z`.
- Immutable revision: `f582d3c922a9863d3ece0e4e6ee40290bdcf68b9`.
- Replay artifact: `replay/native-dtype-evidence.zip`.
- SHA256: `d0916f35313f38da9da61171b03d41abf5ed09c8190fd96729e8acb46bc4682e`.

## Native TRL SFTTrainer LoRA

H200 and Ascend 910C execute native TRL SFTTrainer with aligned TRL 0.29.0, Transformers 5.3.0, PEFT 0.20.0 and Accelerate 1.12.0. The BF16 base stays frozen; FP32 q/v adapters use rank 8, alpha 16 and dropout 0. Every optimizer step accumulates two microbatches with 8 and 12 valid targets and native denominator 20. AdamW uses lr .001, betas (.9,.999), eps 1e-8, no decay, clipping or fused/foreach optimizer. Each requested stage captures all 32 adapter tensors / 1,114,112 elements, including moments and counters.

Step 1 starts from identical stored adapter/optimizer state. Step 2 follows each platform's independent native trajectory. The 427-tensor frozen base / 8,953,803,264 elements is unchanged. All 23 replay checks matched. Full captures retain native losses, per-target scores, complete gradients and stored updates.

- Prefix: `trl-sfttrainer/compare-sft-20261021T111720Z`.
- Final immutable revision: `f333e60d5d399a8fe023f968e6bf11ecbf125177` (`closeout.zip`).
- Comparison delivery revision: `76e0ed69ce2ab1d92bfefea9d0b8ddd6fbd9b089` (`delivery/evidence.zip`).
- Native GPU archive revision: `5b297bf490f30926abff6f6fb99d13e5853c9250`.
- Native NPU archive revision: `6c1ee63d8004ceb76897b0b817b3edd40d8300f8`.

## Native checkpoint restoration

Each platform restores its own checkpoint-1 and executes step 2. This gives three comparisons: H200 continuous/restored, Ascend 910C continuous/restored, and H200/910C restored trajectories. All comparisons retain the same collection settings as the cold SFT experiment.

All 128 parameter/moment/counter entry arrays agree with the corresponding native checkpoint on both platforms. Scheduler output and Trainer step/epoch agree. H200 has 448/448 equal exported arrays; Ascend 910C has 298/448. All 20 target scores agree within each platform; Ascend's native microbatch loss differs by at most `5.960464477539063e-8`. All 39 comparison/manifest-validation replay checks matched.

- Prefix: `trl-sfttrainer-resume/compare-sft-resume-20261021T131901Z`.
- Final immutable revision: `50d2331e7001f54ca0b7409b8f50ee471fe2e973` (`closeout/evidence.zip`).
- Comparison delivery revision: `b586b0a06b281bb01d9032544021faa8e5f3bd03` (`delivery/evidence.zip`).
- Native GPU archive revision: `c83007741d6850dc8761813ca4ffdbd2c2d2f82b` (`gpu-resume-retry/evidence.zip`).
- Native NPU archive revision: `9ce498560dd5d2aef6c41a74de7206f1d2465e80` (`npu-resume/evidence.zip`).

Maximum-error coordinates are retained in the evaluation JSON. Full native artifacts and replay material remain in these immutable archives.
