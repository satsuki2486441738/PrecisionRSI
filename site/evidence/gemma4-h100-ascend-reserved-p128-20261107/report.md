# Gemma-4-12B · H100 and Ascend · Reserved48 P128

This run completed the P128 portion of the Reserved48 evaluation on NVIDIA H100 and Ascend. Both backends used the same checkpoint, BF16 parameters and logits, eager attention, seed 0 and tensor parallel size 1. The prompts used 12 WikiText-2 English training-reserved articles with 128 prompt tokens and 32 fixed continuation tokens each. The three execution modes were cache-off full, prefill followed by token decode, and a 63/65-token chunked prefill followed by token decode.

The analysis covers 36 matched cross-backend and 72 within-backend article comparisons across nine pair families. Each family pools 1,908 scored target positions, a 384-position continuation subset and 564 selected distribution positions across the 12 articles. Comparisons on the same article share context. The 5,076 distribution comparisons therefore come from 12 independent articles.

Target metrics use source log probabilities pooled by token position. The common CPU independently normalized the selected full-vocabulary BF16 rows in FP64. The source capture retained the remaining rows as scalar evidence and digests, so their full-vocabulary normalizers remain unverified independently. Complement-merged Jensen–Shannon divergence includes the probability mass outside the selected union.

## Paired metrics

Each line gives target MAE in nats, continuation MAE in nats, top-1 agreements over 564 selected positions, and mean complement-merged JS in nats. [Summary JSON](summary.json) preserves all nine families' aggregate scalar metrics and their full numerical precision.

- **GPU cache-off full ↔ NPU cache-off full**. Target MAE 0.03721338110105829, continuation MAE 0.03328602863726362, top-1 550/564, mean JS 0.0003684083820714987.
- **GPU prefill plus token decode ↔ NPU prefill plus token decode**. Target MAE 0.0370568614319282, continuation MAE 0.03729230363814778, top-1 551/564, mean JS 0.0006413757636155694.
- **GPU 63/65 chunked prefill plus token decode ↔ NPU 63/65 chunked prefill plus token decode**. Target MAE 0.03763907662252549, continuation MAE 0.03477081212605979, top-1 559/564, mean JS 0.0005754783749593707.
- **GPU cache-off full ↔ GPU prefill plus token decode**. Target MAE 0.035991346774847055, continuation MAE 0.03507092270392566, top-1 551/564, mean JS 0.0005536833311511724.
- **GPU cache-off full ↔ GPU 63/65 chunked prefill plus token decode**. Target MAE 0.03709565803498355, continuation MAE 0.03558386773207039, top-1 550/564, mean JS 0.00038952233776229687.
- **GPU prefill plus token decode ↔ GPU 63/65 chunked prefill plus token decode**. Target MAE 0.03552430235280898, continuation MAE 0.033243720604726135, top-1 555/564, mean JS 0.0007426657165252134.
- **NPU cache-off full ↔ NPU prefill plus token decode**. Target MAE 0.038111560372121184, continuation MAE 0.037983406219821166, top-1 558/564, mean JS 0.00035662254088549353.
- **NPU cache-off full ↔ NPU 63/65 chunked prefill plus token decode**. Target MAE 0.03685063437538776, continuation MAE 0.03699733495964954, top-1 556/564, mean JS 0.0004628109084588167.
- **NPU prefill plus token decode ↔ NPU 63/65 chunked prefill plus token decode**. Target MAE 0.03785408392221956, continuation MAE 0.03675873276481931, top-1 553/564, mean JS 0.0004384161970175039.

## Scope

Paired coverage is 12 of the 48 planned articles. This report covers fixed-continuation numerical comparisons. Free-generation quality (Q02), inference performance (P01), and independent E02 checks for this P128 result remain outside the measured scope. Observer-transparency E02 applies to its own workload. Numerical acceptance remains unset.

The paired CPU run recorded 810.735335550271 seconds. That interval is instrumented comparison work. The publication summary omits private source paths and observation coordinates.
