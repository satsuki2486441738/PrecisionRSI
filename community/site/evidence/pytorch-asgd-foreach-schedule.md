# PyTorch ASGD foreach learning-rate schedule backport

2026-10-22 · historical v2.3.0 ASGD source executed on installed Torch 2.6.0 CPU. This is one upstream-fixed defect and a one-expression partial source backport, not a full PyTorch 2.3 installation or a new current-version defect.

## Cause and independent recurrence

The noncapturable foreach path stores `lr / (1 + lambd * lr * step ** alpha)` instead of `lr / ((1 + lambd * lr * step) ** alpha)`. The exponent applies to the whole affine term. The patch changes only these parentheses; dtype, dispatch and allocation policy are unchanged.

A Python scalar recurrence uses ordinary arithmetic and math.pow, with no Torch optimizer helpers. With previous eta/mu, update `x=(1-lambd*eta)*x-eta*g_eff`, then `ax=ax+mu*(x-ax)`, and store the corrected next eta and `mu=1/max(1,step-t0)`. Fixed replay uses retained gradients; the separable-quadratic workload uses native autograd and independently derived analytic oracle gradients.

At lr=0.5, lambd=0.2, alpha=0.75, active step one, original eta is 0.45454545454545453 versus 0.46550622231111144 expected. Eta first diverges at call one; parameters, stored updates and averaging state first diverge at call two when eta is consumed. Quadratic gradients diverge later at call four; call three has no gradients. This distinguishes local schedule error from propagated gradient error.

## Matched native workload and coverage

The byte-identical old ASGD module is loaded under torch.optim and uses installed 2.6.0 optimizer helpers and native tensor operations. Native dispatch remains ASGD.step → asgd → _multi_tensor_asgd. Two real contiguous CPU parameters contain five elements. Each FP32/FP64 fixed-gradient or quadratic trajectory makes 12 native calls, including no-gradient calls three and eight: ten active updates per parameter, not 12 active steps. Zero gradients remain present and advance state.

Four dtype/workload trajectories run uninterrupted or with save/load after call two or five: 12 scenarios, not 12 independent datasets. The same initial fixture and final harness are retained for all sources. Each scenario pairs 302 boundary coordinates plus ten explicitly absent gradient coordinates. Per source there are 3,624 paired coordinates, 144 native calls and 240 active parameter-update events; repeated checkpoint variants/state observations are included. No missing coordinates.

| FP64 fixed-gradient replay, maximum absolute error | Original | Candidate |
|---|---:|---:|
| Stored next-step eta | 0.022730721150762245 | 0 |
| Actual stored update | 0.008969101605693142 | 2.220446049250313e-16 |
| Parameter | 0.00935452104138107 | 2.220446049250313e-16 |

Across all candidate coordinates the maximum is 2.220446049250313e-16 for FP64 and 4.141734866180968e-08 for FP32. Original workload passes 0/12 scenarios; candidate 12/12. Seven permanent regression methods pass after patch; the original run retains 18 failing subtest/assertion records, not 18 methods. Scalar tolerances are 2e-12 (FP64) and 3e-6 (FP32).

Same-source native torch.save/load continuation is exactly equal to uninterrupted boundaries: eight checks per source, 32 across baseline/candidate/v2.4.0-source/installed-2.6.0 controls. Both fixed controls pass 12/12 oracle scenarios. Original alpha=1, lambd=0 and single-tensor controls pass; alpha=0 is another failing schedule case. Regression also covers nearby alpha values, maximize and weight decay.

## Source provenance and limits

- Affected v2.3.0 source: `97ff6cfd9c86c5c09d7ce775ab64ec5c99230f5d`, torch/optim/asgd.py.
- Initial fix `2c5ad9a3d7ea79ca897aec153a401f4b9175a717` (May 15, 2024) was reverted by `e3c5d1b7d7645e3e9c9efc71cbbaa77248fda9c2` for lint failure (May 16).
- Landed fix [`7e166e805740766268b4297621961a4afdd6c41b`](https://github.com/pytorch/pytorch/commit/7e166e805740766268b4297621961a4afdd6c41b), PR #126375, May 17, 2024. The local backport intentionally implements only the noncapturable schedule expression, not the whole commit.
- Verified fixed v2.4.0 source: `d990dada86a8ad94882b5c23e859b88c0c255bda`, executed on the same Torch 2.6 runtime. No exhaustive affected-version or first-fixed-release claim.
- Pinned main `37f56e48ce431907080585a595c437f9edb23411` was inspected, not executed.

Scope is CPU FP32/FP64 noncapturable foreach with synchronized per-parameter steps. The separate unequal-step defect and capturable branch are unchanged and unvalidated. No GPU/NPU, compiled, complex, sparse, noncontiguous, distributed or differentiable coverage. Checkpoint checks are same-source continuation; already incorrect eta checkpoints are not migrated. No performance claim. Full artifacts stay with the private case; public JSON retains the initial fixture, measured errors, provenance and hashes. Numerical acceptance awaits user judgment.

[Measurements and provenance](pytorch-asgd-foreach-schedule.json) · [One-expression backport](pytorch-asgd-foreach-schedule.patch).
