# GPU and Ascend model precision evaluation

Snapshot 2026-11-12T17:28:08.162990+00:00. PrecisionRSI tools at `118b50032dabd6f771b3a8391e24cef1d02e9a1a`.
The native and reference peers run on the same assigned accelerator set within each platform. The comparison uses the same checkpoint revisions and frozen inputs across the campaign.
Numerical acceptance is undetermined because the protocol supplied no tolerance.

| Model | CUDA status | CUDA held-out ΔNLL | Ascend status | Ascend held-out ΔNLL |
| --- | --- | --- | --- | --- |
| Gemma 3 27B IT | completed | +0.003210 [-0.003743, +0.010252] | completed | -0.007520 [-0.022117, +0.007378] |
| gpt-oss-120b | completed | +0.007730 [-0.005522, +0.021593] | completed | +0.008509 [-0.006256, +0.022890] |
| Nemotron 3.5 Bolt 74B-A9B BF16 | completed | +0.001148 [-0.000526, +0.002727] | completed | -0.000511 [-0.004032, +0.002885] |

ΔNLL is mean native minus reference in nats per token over 48 reserved requests, with a length-stratified 95% bootstrap interval in brackets.

## Recorded measurements

| Platform and model | Devices per pair | Discovery full targets | Logprob MAE | P99 | GSM8K native / ref | Device hours |
| --- | --- | --- | --- | --- | --- | --- |
| cuda · Gemma 3 27B IT | 1 × NVIDIA H20Z | 11,124 | 0.0726018 | 0.743074 | 9/12 vs 8/12 | 1.185 |
| cuda · gpt-oss-120b | 4 × NVIDIA H20Z | 11,124 | 0.212101 | 1.93763 | 8/12 vs 8/12 | 15.401 |
| cuda · Nemotron 3.5 Bolt 74B-A9B BF16 | 2 × NVIDIA H20Z | 11,124 | 0.0513824 | 0.458249 | 5/12 vs 6/12 | 41.021 |
| ascend · Gemma 3 27B IT | 2 × Ascend910_9382 | 11,124 | 0.0651031 | 0.613769 | 8/12 vs 9/12 | 3.930 |
| ascend · gpt-oss-120b | 2 × Ascend910_9382 | 11,124 | 0.194361 | 1.74814 | 9/12 vs 8/12 | 60.336 |
| ascend · Nemotron 3.5 Bolt 74B-A9B BF16 | 2 × Ascend910_9382 | 11,124 | 0.0960378 | 0.727686 | 5/12 vs 6/12 | 33.512 |

Device hours cover closed intervals at the recorded audit times. The paired measurement files include sampled vocabulary agreement, every completed text mode, operator and modality coverage, paired quality and performance, and source hashes.


## Backend configurations

- cuda · gpt-oss-120b. Both peers use the same original MXFP4 checkpoint bytes. The 32K runs use 256-query blocks of the owning eager attention with sink semantics.
- ascend · Gemma 3 27B IT. The 32K path projects logits in 256-row tiles and retains all valid target positions.
- ascend · gpt-oss-120b. Both peers use the same original MXFP4 checkpoint bytes. Ascend executes BF16 expert arithmetic. The 32K runs use 256-query blocks of the owning eager attention with sink semantics. At 8192 tokens, eager attention used 256-query blocks after the original full attention exceeded device memory. The 32K path projects logits in 256-row tiles and retains all valid target positions.
- ascend · Nemotron 3.5 Bolt 74B-A9B BF16. Accelerate streams offloaded weights to the assigned NPUs. The Torch Mamba recurrence retains the released multiply-and-sum axes in bounded tiles. The short same-role raw logits matched byte for byte. The 32K run frees Mamba temporaries after their last use and sets a 44 GiB model-weight limit on each NPU for both peers. Its reference attention uses 256-query blocks. At 8192 tokens, eager attention used 256-query blocks after the original full attention exceeded device memory. The 32K path projects logits in 256-row tiles and retains all valid target positions. Discovery uses released expert dispatch. Later stages skip empty bias-free experts. Same-input paired pilots preserved every compared logit and cache byte.

## Replay and provenance

The archive stores anonymized rows `[target_count, native_nll, reference_nll]` for each paired discovery, reserved, or long full-forward stage. For each peer, sum `target_count × nll` and divide by total target count to reproduce the recorded pooled NLL. Average the per-sample peer differences across the four reserved bins to reproduce the 48-input mean. [Machine-readable results](results.json)
The frozen protocol SHA-256 is `95d0e9bea53d9ee11c23b383def33b6af4bd05bea3741aab65f64ae6e66dfa31`. Checkpoints, inputs, full logits, traces and connection details are retained in the private execution stores. The archive records their audit digests where available.
