# PEFT DoRA Conv1D unmerge orientation — round 3

2026-10-22 · installed PEFT 0.17.1, Torch 2.6.0, Transformers 4.53.3 · CPU FP32/FP64. Independently rediscovered historical defect, fixed upstream; verified isolated local backport. This is distinct from the earlier LoRA SVD scaling repair.

## Cause and independent computation

Transformers Conv1D stores affine weights as (input, output). DoRA merge orients its per-output magnitude factor with fan_in_fan_out, but unmerge in 0.17.1 always divides rows. Square weights silently accept the wrong axis; rectangular weights raise a dimension mismatch. Conv1D here is transformers.pytorch_utils.Conv1D, not torch.nn.Conv1d.

The independent dense oracle computes D=(alpha/r)*A.T@B.T, V=W+D, n[j]=sqrt(sum_i V[i,j]^2), f[j]=magnitude[j]/n[j], and Y=X@(V*f[None,:])+bias. Merge stores Wm=V*f[None,:]; inverse is W=Wm/f[None,:]-D. It uses no PEFT norm/delta/merge helper. Training follows DoRA's detached-norm gradient convention. The two-line patch orients the inverse factor exactly as merge already does; dtype, precision and checkpoint policy are unchanged.

## Matched native verification

Native PeftModel, one default adapter, rank 3, nonzero A/B, magnitudes 0.4–1.8, configured alpha 1/7/13. Thirty-two combinations cover DoRA/LoRA, Conv1D/Linear, square/rectangular storage, FP32/FP64 and safe/ordinary merge, plus two alpha cases. Base storage shapes are 5×5 and 7×5. Baseline and candidate consume the same immutable 34-configuration fixture and identical final permanent test bytes. Native call traces verify the actual unmerge implementation executed.

| Primary FP64 square maximum absolute error | Original | Backport |
|---|---:|---:|
| Output after unmerge | 2.479339473837877 | 8.881784197001252e-16 |
| Restored base weight | 2.161740176813238 | 2.220446049250313e-16 |
| B gradient after three cycles | 4.371820268877045 | 8.881784197001252e-16 |
| Actual B SGD update | 0.04371820268877044 | 2.7755575615628914e-17 |

Original: 24 controls pass, 10 affected configurations fail (six silent square corruptions, four rectangular exceptions). Backport: 34/34 pass. Successful transitions check three merge/unmerge cycles, unchanged bias and adapter tensors, disabled-base output, resumed training/input and adapter gradients, direct SGD lr=0.01, post-update output, native safetensors adapter reload on the original CPU base and another cycle. Checkpoint checks establish persistence after the same repair, not a second defect.

Across the candidate suite, unmerged-output maximum error is 1.3322676295501878e-15 in FP64 and 4.76837158203125e-7 in FP32. Absolute tolerances are 3e-12/3e-5, multiplied by five for training and post-update values. Numerical acceptance remains pending user judgment.

## Provenance and scope

Failure, hypothesis and red/green verification precede upstream lookup; the retained chronology hash is in the JSON. Upstream PR [#3532](https://github.com/huggingface/peft/pull/3532) merged August 14, 2026 as 607895c9907936b296598d2a6520a9e5a74f82a5. Inspected main a8b13931ca7e44f5c0c428cf33e785ad2e8d51fa has the same transpose correction. Current main was inspected, not executed. No local source commit exists; no novelty or earliest-fixed-release claim.

Scope is tiny unquantized CPU affine layers with one active adapter and finite nonzero factors, no dropout. GPU/NPU/MPS, multiple active adapters, zero magnitude, save while merged and full pretrained workloads are untested. No performance claim. Full fixture, execution logs and original indexed bundle remain private; public JSON retains their source identity and measurement precision.

[Structured evidence](peft-dora-conv1d-unmerge-round3.json) · [Minimal patch](peft-dora-conv1d-unmerge-round3.patch).
