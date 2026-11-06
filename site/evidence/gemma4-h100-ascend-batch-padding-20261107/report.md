# Gemma-4-12B H100–Ascend batch and padding comparison

This measured I04 stage uses one frozen English discovery parent and its 96-token prefix as shape stress. A has 160 tokens/159 targets; B has 96 tokens/95 targets. Each backend loads one model and performs four cache-off native calls, producing six logical observations. Parameters and retained native logits are BF16; exported log probabilities are FP64.

The independent admission checked all 572 native files, 1536 valid predictor rows and 564 complete vocabulary rows. All 1792 physical predictor rows are retained; 256 pad-only rows are counted separately and never normalized or paired. Finite-value assertions apply to valid rows.

The 18 request-preserving contrasts contain six matched GPU/NPU pairs and twelve within-backend controls. Their full scalar results and 846 distribution-row contrasts are archived without pooling unequal target counts. Top100-union probability errors and exact-complement-merged JS retain their coarsened definitions.

- gpu-alone-a-a-vs-npu-alone-a-a: 159 targets; MAE 0.033320404862961756 nat, P99 0.1683911049445629 nat, maximum 0.18670247219334346 nat at anonymous target index 56.
- gpu-batch-left-a-vs-npu-batch-left-a: 159 targets; MAE 0.03481040934801931 nat, P99 0.13804556883236646 nat, maximum 0.3398118101119194 nat at anonymous target index 96.
- gpu-batch-right-a-vs-npu-batch-right-a: 159 targets; MAE 0.03481040934801931 nat, P99 0.13804556883236646 nat, maximum 0.3398118101119194 nat at anonymous target index 96.
- gpu-alone-b-b-vs-npu-alone-b-b: 95 targets; MAE 0.03505970429405074 nat, P99 0.1551944103931855 nat, maximum 0.2478476750605889 nat at anonymous target index 43.
- gpu-batch-left-b-vs-npu-batch-left-b: 95 targets; MAE 0.03206222864149264 nat, P99 0.12977852892511055 nat, maximum 0.15136468868294628 nat at anonymous target index 56.
- gpu-batch-right-b-vs-npu-batch-right-b: 95 targets; MAE 0.03206222864149264 nat, P99 0.12977852892511055 nat, maximum 0.15136468868294628 nat at anonymous target index 56.

Actual independent E02 replay matched all 18 target reductions and 283 selected distribution contrasts, including all 282 matched cross-backend rows and the deduplicated reported extrema. It freshly hashed all 572 native files. This does not independently repeat all 846 distribution contrasts or all valid native rows.

The public standalone replay opens only the two frozen JSON files and recomputes all 18 target comparisons with the Python standard library:

```sh
python -S -B replay-targets.py selected-targets.json selected-metrics.json
```

Recorded model-child wall times include imports, model loading, instrumentation, capture and validation. They are a subset of execution accounting, with no speed ratio, synchronized latency/throughput, allocated-device lifetime or billing conclusion. P01 and the full core remain incomplete.

Thresholds and acceptance remain null. Backend/operator causes remain unresolved. Coverage refers to the exact measured stage; one discovery parent does not establish representative quality or multilingual coverage.

Artifacts: [targets](selected-targets.json), [metrics](selected-metrics.json), [independent checks](independent-checks.json), [execution accounting](execution-cost.json), [plan coverage](coverage.json), [replay](replay-targets.py).
