# GPU and Ascend model precision evaluation

Snapshot 2026-11-11T10:44:41.707163+00:00. PrecisionRSI tools at `118b50032dabd6f771b3a8391e24cef1d02e9a1a`.
The native and reference peers run on the same assigned accelerator set within each platform. The comparison uses the same checkpoint revisions and frozen inputs across the campaign.
Numerical acceptance is undetermined because the protocol supplied no tolerance.

| Model | CUDA status | CUDA held-out ΔNLL | Ascend status | Ascend held-out ΔNLL |
| --- | --- | --- | --- | --- |
| Gemma 3 27B IT | completed | +0.003210 [-0.003743, +0.010252] | completed | -0.007520 [-0.022117, +0.007378] |
| gpt-oss-120b | completed | +0.007730 [-0.005522, +0.021593] | partial | Pending |
| Nemotron 3.5 Bolt 74B-A9B BF16 | completed | +0.001148 [-0.000526, +0.002727] | partial | Pending |

ΔNLL is mean native minus reference in nats per token over 48 reserved requests, with a length-stratified 95% bootstrap interval in brackets.

## Recorded measurements

| Platform and model | Devices per pair | Discovery full targets | Logprob MAE | P99 | GSM8K native / ref | Device hours |
| --- | --- | --- | --- | --- | --- | --- |
| cuda · Gemma 3 27B IT | 1 × NVIDIA H20Z | 11,124 | 0.0726018 | 0.743074 | 9/12 vs 8/12 | 1.185 |
| cuda · gpt-oss-120b | 4 × NVIDIA H20Z | 11,124 | 0.212101 | 1.93763 | 8/12 vs 8/12 | 15.401 |
| cuda · Nemotron 3.5 Bolt 74B-A9B BF16 | 2 × NVIDIA H20Z | 11,124 | 0.0513824 | 0.458249 | 5/12 vs 6/12 | 41.021 |
| ascend · Gemma 3 27B IT | 2 × Ascend910_9382 | 11,124 | 0.0651031 | 0.613769 | 8/12 vs 9/12 | 3.930 |
| ascend · gpt-oss-120b | 2 × Ascend910_9382 | 11,124 | 0.194361 | 1.74814 | Pending | 10.420 |
| ascend · Nemotron 3.5 Bolt 74B-A9B BF16 | 2 × Ascend910_9382 | 11,124 | 0.0960378 | 0.727686 | Pending | 14.290 |

Device hours cover closed intervals at the recorded audit times. The paired measurement files include sampled vocabulary agreement, every completed text mode, operator and modality coverage, paired quality and performance, and source hashes.

- ascend · gpt-oss-120b pending tasks D01, I01, I02, I03, I06, I05, O03, O04, O05, O06, Q01, Q02, P02, E01, E02, E03, X01, A-MOE02, Z02, E04.
- ascend · Nemotron 3.5 Bolt 74B-A9B BF16 pending tasks D01, I01, I02, I03, I06, I05, O03, O04, O05, O06, Q01, Q02, P01, P02, E01, E02, E03, X01, A-MOE02, A-MM02, E04.

## Replay and provenance

The [machine-readable archive](results.json) stores anonymized rows `[target_count, native_nll, reference_nll]` for each paired discovery, reserved, or long full-forward stage. For each peer, sum `target_count × nll` and divide by total target count to reproduce the recorded pooled NLL. Average the per-sample peer differences across the four reserved bins to reproduce the 48-input mean.
The frozen protocol SHA-256 is `95d0e9bea53d9ee11c23b383def33b6af4bd05bea3741aab65f64ae6e66dfa31`. Checkpoints, inputs, full logits, traces and connection details are retained in the private execution stores. The archive records their audit digests where available.
