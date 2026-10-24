# Qwen3.5-9B-Base native pretraining token normalization repair

Case ID: `transformers-530-qwen35-pretraining-token-normalization`.
This repairs a defect present in the installed upstream Transformers **v5.3.0** source. It is not a historical backport or an injected bug. The verified source commit is `aad13b87ed59f2afcfaebc985f403301887a35fc`. This claim concerns that installed revision, not every newer upstream version. No PEFT adapters are involved.

## Model, workload and limits

The complete declared `Qwen/Qwen3.5-9B-Base` checkpoint at revision `68c46c4b3498877f3ef123c856ecfde50c39f404` was verified again against the retained upstream per-file digests. Four full weight shards, 775 indexed tensors and 9,653,104,368 checkpoint parameters passed verification. The unchanged native `Qwen3_5ForConditionalGeneration` architecture has 32 text layers and loads 9,409,813,744 parameters with empty missing/unexpected/mismatched-key diagnostics. The separate header inventory accounts for the checkpoint's auxiliary MTP tensors; this is native loading behavior, not a case-defined smaller configuration. The vision tower is loaded but unused in text-only pretraining. No random initialization, layer truncation or smaller model configuration is used.

The original task is causal next-token pretraining: all real text tokens are labels. Two fixed plain-English texts tokenize to lengths 6 and 31, giving 5 + 30 = 35 effective next-token targets. Exact texts and token IDs are retained in the successful native result. Batch size is 1, gradient accumulation is 2, weights and gradients use BF16, attention uses native eager attention, cache is disabled, and unavailable optional FLA/causal-conv kernels cause Transformers to select its own Torch fallback. All text-model parameters participate in backward; gradient evidence covers the full last-layer down-projection matrix and the output-head rows for the 31 distinct input/target IDs.

Both native and reference backward use the same `Accelerator.backward`, including its configured accumulation scaling of 2. The independent reference changes only the objective reduction to summed next-token CE / 35. Thus the gradient comparison isolates objective normalization under identical engine scaling; it does not independently certify the engine's absolute gradient scale. **No optimizer step was executed**, and optimizer updates, convergence, multimodal training, distributed training and all model gradients are outside the verified scope.

Runtime: Python 3.12.13, Torch 2.9.0, torch-npu 2.9.0.post2, Transformers 5.3.0, Accelerate 1.14.0, CANN toolkit/A3 ops 9.1.0 (`V100R001C11SPC001B243`), device `Ascend910_9382`. Only physical device 12 was exposed, as logical device 0. Every native launch saved and checked current device occupancy. The selected NPU was empty before each launch. Successful patched launch took 32.986 s including process startup; collector wall time was 13.982 s and peak NPU allocation was 42,891,059,200 bytes (39.945 GiB).

## Cause and isolated patch

In upstream `Qwen3_5ForConditionalGeneration`, `accepts_loss_kwargs=False` prevents Trainer from passing the accumulation-window target count. Its forward also omits that denominator when calling its causal loss. Trainer consequently averages the two microbatch means equally. For these unequal lengths this optimizes `(mean_CE_5 + mean_CE_30)/2` instead of `(sum_CE_5 + sum_CE_30)/35`.

Enabling the token denominator also exposes Trainer's independent alignment error: its default count includes every nonignored label, but causal loss drops each sequence's first label. This workload would count 37 instead of 35. The patch therefore (1) enables and forwards `num_items_in_batch` in the native conditional-generation class and (2) counts shifted causal targets in Trainer. Explicit `shift_labels` are honored. Masked LM, custom-loss and label-smoother counting retain their existing behavior. Both patched files live only in the case's remote `vendor/transformers/` copy; shared libraries are unchanged. The complete diff is `evidence/trainer-target-count.patch`.

## Measured results

| Measurement | Before | After | Independent reference |
| --- | ---: | ---: | ---: |
| Accumulated pretraining loss | 2.571202278137207 | 2.021854519844055 | 2.021854519844055 |
| Absolute loss error | 0.549347758293152 | 0 | — |
| Relative loss error | 27.1704889200% | 0 | — |
| Effective Trainer denominator | absent; microbatch averaging | 35 | 35 |
| Compared gradient elements | 50,458,624 | 50,458,624 | same coordinates |
| Gradient L2 error | 9.956697922364665 | 0 | — |
| Gradient relative L2 error | 1.2311948196700295 | 0 | — |
| Gradient cosine | 0.6879998693373885 | 0.9999999999999998 | — |
| Gradient maximum absolute error | 1.076171875 | 0 | — |

Patched gradients are exactly equal in all retained coordinates; there are no missing arrays or nonfinite values. Arithmetic and coverage come from the existing precision-compare `array_compare.compare_arrays`, reused without editing it. Acceptance checks used loss absolute tolerance 2e-5 and gradient `rtol=0.02, atol=1e-6`, allowing BF16 rounding; the observed patched difference is exactly zero.

Eight CPU regressions call the actual Trainer counting method. Four causal cases fail before repair; all eight pass afterward, including masked/custom/smoother controls. These fixtures create no model and inject no defective implementation. `finish.py` checks retained native before/after evidence and CPU red/green results without another model execution.

## Execution evidence and failures

Exactly four native launches were used, each with a 900-second timeout; all finished in under 33 seconds. This preserves both setup failures rather than presenting them as numerical evidence:

| Remote run | Duration | Result |
| --- | ---: | --- |
| `runs/npu-before-20261024T142035Z` | 23.915 s | Inherited `LOCAL_RANK` was a comma-separated device list; Accelerate rejected it after model loading. |
| `runs/npu-before-20261024T142110Z` | 25.111 s | The original candidate assumed loss kwargs were enabled; native model flag inspection disproved that trigger. |
| `runs/npu-before-20261024T142437Z` | 31.418 s | Original forward/backward, oracle and gradient export completed; final JSON serialization failed on sets in loading diagnostics. |
| `runs/npu-after-20261024T143005Z` | 32.986 s | Patched original workload and native regression passed, exit 0. |

The numerical red evidence is the third run's existing log and gradient arrays/comparison, together with the failing CPU regression. Because its final JSON write failed, `recovered-before.json` explicitly reconstructs only values already present in those artifacts. It does not reconstruct missing timing or memory information. Original run bytes remain unchanged. The collector now normalizes the loading-diagnostic sets for serialization. Remote source-download timeout during preparation was resolved by transferring the exact upstream source; it was not another native execution.

Local `evidence/EXPORT.json` records the hashes and sizes of the 29 fetched evidence files. Large NPY gradient arrays and the isolated source package remain remote. Local `checkpoint-evidence/EXPORT.json` indexes the supplemental header inventory. `FILES.md` lists local additions.

Remote work root:

```
/home/work/user-job-dir/app/slai/user-xiexinyuan/xinyuan/precision-skill/precision-skills-publish/projects/precision-repair/cases/pretrained-training-repair-20261024T141838Z
```

The measured baseline is `runs/npu-before-20261024T142437Z/artifacts/`; the successful repair is `runs/npu-after-20261024T143005Z/artifacts/`. Each contains native Trainer and independent-oracle gradient NPYs plus `gradient-comparison.json`. The latter also contains `result.json`. Every run has `command.json`, `device-before.txt`, stdout and stderr. Model resources remain in their original read-only location. No shared skill, repository tests, site histories or installed libraries were edited.

## Reuse and actual workflow improvements

The requested precision-repair skill supplied causal diagnosis and original-workload verification. precision-compare supplied `record_run`, `compare_arrays` and bounded hash-verified `evidence_bundle.fetch`.

Actual reusable improvements are: inspect a model's real `accepts_loss_kwargs` flag and loss forwarding before an expensive launch; validate inherited distributed rank variables in a single-device adapter; and normalize sets in native loading diagnostics before JSON serialization. The first two setup omissions consumed native budget, and the serialization omission prevented the baseline result JSON despite completed numerical captures. These are evidence-backed adapter/preflight improvements. No defect in the shared precision-compare tools was established, and no changes to shared tools or skills are proposed as already implemented.

For read-only result review, use the retained summary and integrity index. On the remote case, `python finish.py` validates existing red/green evidence without native execution. Do not launch another native run under this task: its four-launch budget has been exhausted.
