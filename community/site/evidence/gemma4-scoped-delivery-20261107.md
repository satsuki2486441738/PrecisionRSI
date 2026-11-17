# Gemma-4-12B H100 and Ascend precision delivery

This delivery measures the text path of `google/gemma-4-12B` base on NVIDIA H100 and Ascend. The paired native stages use matched fixed inputs and one model device per backend (TP1). Backend software and math policies are recorded with each result. Thresholds and numerical acceptance remain unset.

| Evidence | Measured scope |
| --- | --- |
| I01 cache-off anchor | One 160-token input and one full forward per backend. The paired comparison covers 159 targets and 47 selected distributions. Target logprob MAE was 0.033320 nat, and sampled Top-1 agreement was 47/47. Six historical cached contrasts cover only the 32 overlapping continuation targets. [Report](gemma4-h100-ascend-cacheoff-20261106/report.md) |
| I03 partitions and repeats | Three full forwards, token decode, and unequal-chunk prefill followed by decode in one loaded process per backend. The report contains 23 score/distribution comparisons and four final K/V comparisons. All six within-backend full-repeat comparisons reported zero target and sampled-distribution differences. [Report](gemma4-h100-ascend-partitions-20261107/report.md) |
| I04 batch and padding | One 160-token discovery input and its 96-token prefix, compared alone and in left- and right-padded batches. The 18 contrasts preserve request identity, including six matched H100–Ascend pairs. The 256 pad-only rows were excluded from normalization and comparison. [Report](gemma4-h100-ascend-batch-padding-20261107/report.md) |
| H100 observer snapshot | Four cache-off collector settings in one loaded process. The H100 BF16 logits payloads were bit-identical, as were the two enabled modes' retained boundary sample payloads. The snapshot records the original Ascend attempt ending at the shared-instance guard before model load. [Report](gemma4-h100-observer-transparency-20261107/report.md) |
| H100–Ascend observer transparency | Four cache-off observer settings per backend on the same fixed input. Within-backend raw BF16 payload equality held on both devices. Each matched mode agreed at all 47 sampled Top-1 positions. The hooks-only pair's target logprob MAE was 0.033320 nat and sampled boundary RMSE was 0.021232. Independent CPU checks replayed 752 sampled distribution comparisons and 392 sampled boundary comparisons. [Report](gemma4-h100-ascend-observer-transparency-20261107/report.md) |
| Reserved48 H100 partition comparisons | 36 of 48 planned reserved articles across prompt lengths 128, 512 and 2,048. The 108 within-H100 contrasts cover 5,076 sampled distribution pair positions. Both 8,192-token attempts ended before numerical admission. [Report](gemma4-h100-reserved48-three-bins-20261107/report.md) |
| Reserved P128 H100–Ascend partition comparisons | 12 paired reserved articles, each with 128 prompt and 32 forced continuation tokens. Three matched cross-backend modes contributed 36 cross-backend contrasts; 72 within-backend contrasts provide context. Sampled Top-1 matched at 550/564 (full), 551/564 (token) and 559/564 (chunk). [Report](gemma4-h100-ascend-reserved-p128-20261107/report.md) |

The dual observer evaluation record is `gemma4-h100-ascend-observer-transparency-20261107`. Its coverage preserves the original H100-only snapshot and adds paired evidence for M01, M03, D03, O01 and O02. The public replay scripts reproduce target summaries from the supplied JSON. The observer reports separately document sampled boundary values and frontend operator metadata, with kernel attribution and complete tensor trajectories open. [Dual observer coverage](gemma4-h100-ascend-observer-transparency-20261107/coverage.json)

The source release contains indexed, sanitized source and frozen public token inputs. Native entry points are disabled in the released copy. The release README documents verification, source provenance and licensing. [Source ZIP](gemma4-native-precision-source-20261107/source-release-v1.zip) · [Source index](gemma4-native-precision-source-20261107/source-release-v1.index.json) · [Release notes](gemma4-native-precision-source-20261107/README.md) · [CPU source supplement](gemma4-native-precision-source-20261107/source-supplement-v1.zip) · [Supplement index](gemma4-native-precision-source-20261107/source-supplement-v1.index.json)

## Coverage of the original plan

The observer milestone ledger tracks 49 task IDs, including 28 core tasks. Completion in this table applies only to the stated pilot scope. The Reserved48 results add partial evidence for I05 and Q01. The table below preserves the earlier observer milestone; the newer P128 results do not complete either task. All 49 full-plan completion flags remain false. [49-task ledger and scope definitions](gemma4-h100-ascend-observer-transparency-20261107/coverage.json)

| Status | Count | Task IDs |
| --- | ---: | --- |
| Scoped complete | 9 | M02, D01, D02, I01, I02, I03, I04, E01, E02 |
| Partial | 9 | M01, M03, D03, O01, O02, Q01, P02, E03, E04 |
| Pending | 8 | I05, I06, O03, O04, O05, O06, Q02, P01 |
| Deferred | 21 | I07, S01, S02, S03, S04, S05, S06, S07, S08, R01, R02, R03, R04, R05, X01, X02, Z01, Z02, A-MM01, A-MM02, Q03 |
| Not applicable to the pinned non-routed text configuration | 2 | A-MOE01, A-MOE02 |

Representative and long-context coverage, free-generation quality, matched performance, and full resource accounting remain open. Training, RL, distributed layouts, quantization and non-text execution remain deferred. Ascend reserved P512, P2048 and P8192 remain unpaired. Both H100 P8192 attempts stopped before numerical admission; their interruption cause remains unknown. The tiny CPU prerequisite for Q02 stopped at its effective-memory gate before model execution. No claim is made about complete 48-article quality, P01 performance, 4-GPU scaling or 8-device distributed execution.
