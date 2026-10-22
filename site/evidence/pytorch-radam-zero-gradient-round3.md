# PyTorch RAdam foreach zero-gradient branch — round 3

2026-10-22 · installed Torch 2.6.0 (2236df1770800ffea5697b11b0bb0d910b2e59e1), Python 3.12.14 · CPU FP32/FP64. Independent native finding and isolated six-line candidate; novelty unproven, no upstream submission.

With eps=0, the finite unrectified branch (rho_t<=5) should apply momentum without adaptive division. Native foreach evaluates an unused adaptive factor; zero variance creates NaN that contaminates the parameter. First probe: parameter 1 with present zero gradient becomes NaN, while native single-tensor and scalar oracle retain 1; local step advances and moments remain zero. Absent gradients instead skip initialization and step. The candidate zeros only the unused adaptive intermediate before adding momentum, without changing epsilon or skipping a present-zero transition.

The independent Python scalar recurrence branches before division and consumes identical dtype-rounded gradients. Analytic affine/quadratic native objectives, not another optimizer implementation, establish expected values.

| Matrix versus scalar oracle | Paired coordinates | Original nonfinite mismatches | Candidate |
|---|---:|---:|---:|
| Parameters | 4,480 | 496 | 0 |
| Actual stored updates | 4,480 | 496 | 0 |

Matrix: 32 configurations per side, 14 optimizer calls each, five two-element parameters in two groups; per-parameter gradient-present transitions [14,11,9,11,0]. Regression adds both explicit default dtypes (64 matrix configurations) and boundary/checkpoint controls. Original: six methods with 26 failing subcases; candidate: six methods pass. All 448 gradient and 448 optimizer-state snapshots match exactly; 336 control parameter snapshots are bitwise equal. Candidate finite maximum parameter/update errors are 4.80874513097973e-7 / 1.1809473132728954e-7; finite-subset errors are separate from the nonfinite counts. Tolerances: FP32 3e-6, FP64 2e-12.

The original four-call quadratic/zero-gradient workload also reproduces eight nonfinite parameter and eight update coordinates out of 64 each, corrected to zero. Real torch.save/load tests include 16 same-mode continuations per side, a healthy native-single checkpoint transferred to foreach, and four original-objective continuations, with exact resumed/uninterrupted boundaries. This does not recover already-NaN parameters.

Bounded post-discovery searches found no matching report; novelty remains unproven. Current raw source matched pinned last-changing revision 9d2b3dd0d265930c916e1431aaaa3a1f3cf127a5 and retains the buffer block, but current upstream was not installed or executed. Finite expectations exclude rectified rho_t>5 with v=eps=0, which is mathematically undefined. Capturable is unchanged/untested; GPU/NPU, compile, differentiation through optimizer, complex/sparse and large models are untested. Numerical acceptance awaits user judgment. Full logs and indexed ZIP remain private; public JSON retains source/test/bundle hashes.

[Measurements and provenance](pytorch-radam-zero-gradient-round3.json) · [Six-line candidate](pytorch-radam-zero-gradient-round3.patch).
