# Transformers Trainer label-smoothing historical backport

2026-10-22 · Transformers 4.53.3 · native CPU FP32 `Trainer.train()`. This reproduces an upstream-fixed defect and backports the two responsible source files; it is independent of the Qwen2 mask repair.

## Cause and independent objective

Trainer sees loss-kwargs support and skips its legacy accumulation-step division, but LabelSmoother receives no full-window target count and averages each microbatch locally. Four equal-sized microbatches therefore accumulate four times the intended gradient; unequal target counts also alter relative target weighting.

The oracle explicitly constructs `q(k)=(1-epsilon)*1[k=y]+epsilon/V`, sums dense-target losses over nonignored shifted targets and divides by the full optimizer-window count. A separate full-window forward/backward and SGD trajectory consumes the actual captured native sampler windows; it does not use Trainer loss, LabelSmoother or model-internal loss. The noncausal utility additionally agrees with PyTorch cross_entropy with label smoothing and ignored targets.

## Same native workload

A random one-layer Llama has 968 parameters, CPU FP32 eager attention, dropout zero and seed 813. Eight length-five examples run with batch size one, gradient accumulation four, smoothing 0.1, constant SGD lr=0.03 and clipping disabled, for two actual optimizer steps. Every initial array, input window and independent oracle trajectory matches before/after.

| Whole two-step trajectory error | Original | Backport |
|---|---:|---:|
| Gradient maximum absolute error / 1,936 coordinates | 1.353308618068695 | 5.960464477539063e-08 |
| Gradient RMSE | 0.1255671815397329 | 2.724049042635211e-09 |
| Stored-update maximum absolute error / 1,936 coordinates | 0.04059925675392151 | 1.862645149230957e-09 |
| Stored-update RMSE | 0.003767015373292169 | 1.4655550858266856e-10 |

No missing gradient/update coordinates. Stored updates are computed from promoted stored before/after parameters. Native reported training loss is 11.601189613342285 before and 2.940302848815918 after; independent mean step loss is 2.9403027296066284. Trajectory comparisons include second-step state propagation; independent recomputation at each affected pre-step state also measures local gradient-norm ratios near four.

Eight permanent test methods: original five failing methods (four native and one utility), three passing controls; patched eight passing methods. There are 13 original failure records including subtests, not 13 test methods. Six methods invoke native training for two optimizer steps each. Coverage includes unequal target counts, partial final accumulation windows, batch=2/GAS=2, GAS=1 and no-smoothing controls. Standalone causal utility checks use FP32/FP64 and epsilon 0/0.1/0.7; the noncausal utility tests integer/tensor full-window counts. Gradient/update tolerances are 2e-6/2e-7; final parameters 3e-7 and training loss 2e-6.

## Source and limits

Affected release 4.53.3: `a5923d4de7df2fbd1f373dfcfe983216b79b6937`. Upstream fix [`2da6655ef1cb6f814400b129ff8e2f4d0942963f`](https://github.com/huggingface/transformers/commit/2da6655ef1cb6f814400b129ff8e2f4d0942963f), PR #47261, merged July 22, 2026. Candidate execution is 4.53.3 plus the isolated two-file backport; neither the complete fix revision nor latest release was executed. The patch passes the full-window count to smoothing, retaining standalone local-count fallback; dtype, objective and dependencies are unchanged.

All causal examples ignore their first label, so this baseline's pre-shift count equals active shifted-target count. This is not validation of a broader first-label counting issue. Native scope is CPU FP32 Trainer with constant SGD and clipping disabled. Distributed, GPU/NPU, mixed precision, other optimizers, noncausal native training and the full upstream suite are untested. No performance claim. Complete arrays and execution receipts remain with the private case; public JSON retains source, test and report hashes. Numerical acceptance awaits user judgment.

[Measurements and provenance](transformers-trainer-label-smoothing.json) · [Two-file backport](transformers-trainer-label-smoothing.patch).
