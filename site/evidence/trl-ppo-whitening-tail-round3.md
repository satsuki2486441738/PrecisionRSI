# TRL PPO reward-whitening tail leakage — round 3

2026-10-22 · TRL 0.7.10, Torch 2.2.2, Transformers 4.37.2, Accelerate 0.26.1 · CPU. Independently confirmed historical defect with a one-line local backport. Later v0.15.0 PPO rewrite has an equivalent guard; current-main status is unknown and the first-fix commit is unidentified.

Reward masking occurs before masked whitening with shift_mean=False. Its affine transform revives padded zero rewards; unmasked reverse GAE carries them into selected returns, and global advantage whitening affects other selected tokens. The independent episode-length scalar oracle preserves native unbiased sample variance, epsilon 1e-8 and retained reward mean. A causal replay changes only the post-whitening mask: invalid reward -0.11747631336794973 predicts last-valid return error -0.21762487051412685, matching observed -0.21762487051412682. Appending masked storage also changes original selected results, violating terminal zero-bootstrap invariance. The patch re-masks immediately after whitening without changing selected values, GAE, KL or objective policy.

| Against independent oracle | Original | Candidate |
|---|---:|---:|
| Selected return max absolute error, FP64 helper (6 tokens) | 0.21762487051412682 | 0 |
| Advantage max absolute error, FP64 helper (6 tokens) | 0.20173888740686996 | 2.220446049250313e-16 |
| Parameter-gradient max absolute error, FP32 full step (1,113) | 0.06682920455932617 | 3.5762786865234375e-7 |
| Actual SGD-update max absolute error (1,113) | 0.0006682928651571274 | 3.725290298461914e-9 |

Native supported construction uses local random GPT-2 with value head (one layer, hidden 8, two heads, vocabulary 11, dropout zero) and SGD lr=0.01. Complete PPOTrainer.step covers fixed responses, compute_rewards, compute_advantages, clipped policy/value loss, train_minibatch, Accelerate backward and actual prepared optimizer update. One epoch/minibatch, responses length 2/4, six selected EOS-inclusive tokens in eight response-prefix slots. Queries are one token with ordinary right padding and no custom response masks; the earlier response-mask slicing defect is not exercised.

The scalar oracle independently computes rewards, episode-length GAE, moments, clipped loss and analytical logit/value derivatives; a separate identical model propagates those derivatives through the shared Torch/Transformers Jacobian. It does not reuse TRL numerical helpers. Frozen inputs, old/reference states, all initial parameters and test bytes match before/after. Loss with oracle targets already passes on the original, including nontrivial ratio/clipping controls. Four permanent methods: original/pristine isolated source have three failing methods and one passing control, with four failure records including subtests; candidate four methods pass. Controls cover whitening on/off, widths 4/7, padding invariance and FP64/FP32 loss/output gradients. Tolerances: FP64 helper 1e-10, FP32 loss/gradient 3e-6, stored update 1e-7.

Post-discovery immutable source checks: v0.7.10 matches the tested source; v0.11.4 has the same legacy pattern, inspected only; v0.15.0 rewritten PPO zeros padded rewards after whitening, with different API/configuration constraints, not executed. Legacy main paths and rate limits do not establish current PPO correctness or absence. No novelty or first-fixed-release claim; no local source commit or upstream submission.

Uncovered: generation, multiple PPO epochs/minibatches, stochastic long trajectories, arbitrary masks with holes, accelerator/distributed execution and current TRL runtime. No speed or memory claim. Full frozen state and indexed ZIP remain private; public JSON retains source, test, metric and bundle hashes. Numerical acceptance awaits user judgment.

[Measurements and provenance](trl-ppo-whitening-tail-round3.json) · [One-line boundary patch](trl-ppo-whitening-tail-round3.patch).
