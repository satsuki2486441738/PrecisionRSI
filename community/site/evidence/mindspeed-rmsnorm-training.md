# Historical RMSNorm repair: whole-model training verification

Date: 2026-10-21. This continues the [original RMSNorm repair](mindspeed-rmsnorm.md), using the same model revision, historical stack, frozen GRPO trajectory and Norm patch. No backward or optimizer library code was changed.

## Execution and identity checks

Both NPU TP4 runs complete native backward, clipping and one AdamW step. All ranks succeed and selected devices are idle after exit. All 2,608 initial parameter records match between the two runs. Native execution records identify the selected original/corrected Norm hashes.

Each new run passes a source-side audit of 7,768 tensor files. All four ranks record zero nonfinite gradient values. The new runs record effective optimizer groups (lr=0.001, betas=[0.9,0.999], eps=1e-8, weight_decay=0), fresh state and FP32 master parameters.

## Selected tensor comparison

The comparison retains 448 TP rank views of 112 language-model normalization parameters: 1,177,600 scalar observations. Rank views are explicitly counted rather than deduplicated into whole-model logical coordinates. The same historical GPU first-step evidence is used for both comparisons.

| Metric | Before | After |
| --- | ---: | ---: |
| Preclip gradient rmse | 0.0007440127807722619 | 3.8042921997408873e-05 |
| Preclip gradient cosine | 0.07177318770899976 | 0.998692234664261 |
| Preclip gradient max_abs | 0.11377160251140594 | 0.004638671875 |
| Postclip gradient rmse | 1.821202635778471e-05 | 9.359433000968283e-07 |
| Postclip gradient cosine | 0.07177318773893825 | 0.9986922345895994 |
| Postclip gradient max_abs | 0.0027937821578234434 | 0.00011367557090125047 |
| Stored update rmse | 0.0013107354012305518 | 0.00023884099048028886 |
| Stored update cosine | -0.0005040649203209165 | 0.9752871182075522 |
| Stored update max_abs | 0.00390625 | 0.00390625 |

Whole-model preclip gradient norm: GPU 40.75261806198775; NPU before 0.049505140632390976; NPU after 40.59830093383789.

The selected stored-update maximum difference remains 0.00390625. Residual differences remain unattributed. Historical GPU evidence does not record actual optimizer-group settings; this continuation does not establish isolated optimizer implementation agreement. Numerical acceptance remains pending user judgment.

## Run and artifact identities

- Original run: `npu-update-original-20261021T092540Z`.
- Corrected run: `npu-update-fixed-20261021T093550Z`.
- Retained GPU run: `gpu-update-20261017T105353Z`.
- GPU selected bundle SHA256: `2f39b21746449cd96c27010b3fb21a846d2e0d827cc6702bea64bd7f138f7b01`.
- Original NPU selected bundle SHA256: `563401be20b845cd473d34c13d2529407b7d649f17f0dddd12eb6051ef3ff791`.
- Corrected NPU selected bundle SHA256: `ceb0329e6e680e237f9264c245db1847e36ed8a592fa2c42825a822600640d1b`.
- Cumulative NPU device-attempt wall time: 462.3958592605777 / 600 seconds; prior charges retained.
- Three selected bundles total 27383774 bytes. Full tensors remain at source.

[Machine-readable metrics and coverage](mindspeed-rmsnorm-training.json). Full execution evidence remains in the private case archive.
