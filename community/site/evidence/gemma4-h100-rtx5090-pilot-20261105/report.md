# Gemma 4 12B: single-request text inference precision pilot

**Scoped inference pilot, recorded 2026-11-05.**
The completed captures and comparisons below are scoped to one
fixed-history text request. They are not a model-wide numerical acceptance or a
representative benchmark. Numerical thresholds and acceptance are `null`.

## Model and workload

- Base model: [`google/gemma-4-12B`](https://huggingface.co/google/gemma-4-12B/tree/1a2c7729392586f5ecddae8807ee2ef741eabb9a),
  revision `1a2c7729392586f5ecddae8807ee2ef741eabb9a`, model class
  `Gemma4UnifiedForConditionalGeneration`. The retained snapshot verification
  checked all 8 expected files against pinned upstream metadata. The weight file
  SHA-256 was `fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a`.
  The GPU process checked the identity receipt and file sizes, rather than
  rehashing the weight at every load; the runtime configuration commit field was
  null. The receipt alone does not independently prove which bytes were loaded.
- One frozen English-prose discovery example from the
  [`Salesforce/wikitext` WikiText-2 raw train split](https://huggingface.co/datasets/Salesforce/wikitext),
  dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`; dataset card
  license CC BY-SA 4.0. The input SHA-256 was
  `643d29f23d00f97665c5c72d73c63bc20525950b2ca5c50b435140b6066214f0`.
  It contains 128 prompt tokens including BOS and 32 predetermined continuation
  IDs; the example is discovery data, not independent validation data.
- Each capture ran one prefill and 31 cached single-token forwards, yielding 32
  fixed-history target positions and 32 vocabulary distributions. Native logits
  were BF16. The capture derived full-vocabulary log probabilities offline with
  NumPy FP64 log-softmax; the model itself did not execute in FP64. The native
  logits were not retained; the derived full-vocabulary log-probability rows were retained.

## Execution conditions

Three serial, independently started captures completed on each of one NVIDIA
H100 80GB HBM3 and one NVIDIA GeForce RTX 5090. Each capture used one visible
GPU, the same frozen model and input, BF16 parameters, eager attention,
evaluation/inference mode, seed 0, PyTorch `2.7.0+cu128`, Transformers `5.10.1`
and CUDA runtime 12.8. GPU model is an intentional difference; device drivers
and hardware architecture are platform-specific.

The capture explicitly set float32 matmul precision to `highest` and CUDA matmul
TF32 to `false`; cuDNN TF32 was `true`. Deterministic algorithms, deterministic
warn-only, cuDNN deterministic and cuDNN benchmark were `false`;
`CUBLAS_WORKSPACE_CONFIG` was unset and `PYTHONHASHSEED` was `0`. Actual policy
was checked after model loading and after capture. These settings do not establish
a general determinism guarantee.

## Measured comparison

The three H100 captures were compared pairwise using the shared target-score and
distribution comparators. For each of the three distinct pairs, 32/32 target
positions were paired, target log-probability MAE and maximum absolute difference
were `0.0` nats, and top-1 agreed at 32/32 positions. Tail-merged Jensen–Shannon
divergence had mean `0.0` and maximum `0.0`; maximum per-position probability
RMSE was `0.0`. These are the recorded metrics for this request and these runs,
not a claim of determinism for other inputs or executions.

The three RTX 5090 captures were also compared pairwise. Each of the three
pairs had 32/32 paired targets, zero target log-probability MAE and maximum
absolute difference, 32/32 top-1 agreement, zero mean and maximum tail-merged
Jensen–Shannon divergence, and zero maximum per-position probability RMSE.

The first H100 capture and first RTX 5090 capture were compared at 32/32 target
and 32/32 distribution positions. The target log-probability MAE was
`0.03537468167585757` nats, P99 was `0.1444637830610983` nats, and the maximum
absolute difference was `0.14637308806804405` nats at target position 153
(zero-based; prompt positions 0–127). The P99 uses linear interpolation at
`0.99 * (N - 1)`. Mean negative log-likelihood was `2.7127718954732023`
nats/token on H100 and `2.7139227268556265` on RTX 5090. Top-1 agreed at
32/32 positions.

For the two sides' per-position top-100 token union plus one merged tail,
coarsened Jensen–Shannon divergence averaged `0.0002510469876534106` nats and reached
`0.001018400289233065` nats. Probability RMSE, computed over the union tokens
and averaged equally over the 32 positions, was `0.0009674997146439246`.
The 32 distribution prediction positions are 127–158 and predict target
positions 128–159, respectively. These are shared-comparator results for a
single example. An independent CPU calculation of the 32 rows agreed within
1e-12, and all 447 files in the admission inventory matched their recorded
SHA-256 values. The tail-merged divergence is not full-vocabulary JS.
`thresholds: null`; `acceptance: null`.

## Coverage and limits

This pilot covers a single English text request at batch size 1 with a
128-token prompt and fixed continuation history. It does not cover the other
prepared discovery examples, independent held-out or representative validation,
long-context behavior, multilingual, code or mathematics workloads, padding,
other batch sizes, alternate attention implementations, cache-off or chunked
execution, model quality, performance, or image/audio/video paths. H200,
four-GPU and distributed execution were not measured. No layer/operator
attribution or causal diagnosis was established. No numerical equivalence,
accuracy or acceptance conclusion follows from collection status alone.

## Target-score replay

The accompanying `target-logprob-evidence.json` contains only the 32 paired
anonymous target log-probabilities and recorded target metrics. Run
`python replay-target-logprobs.py target-logprob-evidence.json` with Python 3;
the standard-library replay checks coverage, MAE, linear P99, maximum error and
its anonymous pair index, and both NLL values to absolute tolerance `1e-12`.
The evidence JSON SHA-256 is
`d24687dca58736043af41f2e5aa57f2c27dac6d75e59193a0bb2974aadab329f`;
the replay script SHA-256 is
`94ca130a31c3c538025d84f569c344d7a79f75462cc423003ae0bc4225907bba`.
The hashes identify those exact bytes. Pair index 25 corresponds to target
position 153. No acceptance threshold is applied.

Only the target-score metrics are replayable from these files. Top-1 and
coarsened distribution summaries rely on private full-vocabulary derived
log-probability rows and are not reproduced by the target-score replay. Neither
input text nor token IDs, logits or distribution rows are included.
## Evidence provenance

The accompanying target-score replay files form the published, portable evidence for the target metrics only. Full-vocabulary derived rows, source receipts, model weights, inputs and execution logs remain in the private case. Distribution summaries can be independently checked only with those retained private rows. No numerical acceptance threshold was set.
