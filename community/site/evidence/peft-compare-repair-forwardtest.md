# PEFT: comparison-to-repair forward-test

A fresh agent received a precision-compare report, fixed native workload, fixture and PEFT 0.17.1 source, without the previous diagnosis or patch. The parent selected a known historical defect to test skill behavior; this is a follow-up verification, not a new bug discovery.

The agent reproduced native execution, checked the collector and full-rank composition formula, identified duplicate LoRA scaling, and changed only the isolated PEFT helper. It added an independent native-forward regression, patched the implementation, and reran the unchanged original workload through precision-compare.

| Verification | Before | After |
|---|---:|---:|
| Local scenarios passing | 7/26 | 26/26 |
| Original forward max absolute error (35 coordinates) | 2.0363972705233255 | 7.216449660063518e-16 |
| Original basis-forward max absolute error (20 coordinates) | 0.6887567787917122 | 3.3306690738754696e-16 |
| Original input-gradient max absolute error (21 coordinates) | 11.799886212551122 | 2.220446049250313e-15 |

Original-workload coverage is 76/76 coordinates, 3/3 observations. Inputs, initial state, contracts and independent oracle arrays are unchanged; the parent checked 181 protected file hashes. Local scenarios cover FP32/FP64, three matrix shapes, unit/unequal/zero scales, signed coefficients and a concatenation control. The original workload is CPU FP64; these results do not certify other SVD merge algorithms or devices.

The [machine-readable evidence](peft-compare-repair-forwardtest.json) separates local and original-task verification. The [indexed evidence bundle](peft-compare-repair-forwardtest.zip) includes raw before/after arrays, manifests, source identities, fixture, workload, permanent regression, patch and literal native command records. Exported paths use portable labels; the bundle index retains source/export hashes. It replays four cases through installed tools: two tensor comparisons and two NaN/Inf inventories. All 76 coordinates are finite on each side in this case.

From the precision-compare skill directory, replay retained diagnostics with:

```sh
python scripts/replay_bundle.py --bundle /path/to/peft-compare-repair-forwardtest.zip --output-root runs
```

Replay verifies retained measurements; the separately recorded native runs establish the repair. Native run times include imports and are not a performance benchmark. The agent's only minor friction was the installed-package source layout. Existing tensor tooling handled this inference task without a new framework collector or longer task recipe.
