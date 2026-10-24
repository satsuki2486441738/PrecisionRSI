# Full pretrained Qwen3.5 LoRA weighted-adapter repair

Validated on 2026-10-24. Final native job returned 0 and the permanent workload regression passed. Both earlier numerical candidates retain their failed status.

## Workload and correctness criterion

Loaded the complete `Qwen3_5ForConditionalGeneration` checkpoint from the declared Qwen3.5-9B-Base revision `68c46c4b3498877f3ef123c856ecfde50c39f404`: **9,409,813,744 parameters**, all 32 text layers, original configuration and vision weights retained. The task is text causal inference after a real three-step LoRA SFT. Only `model.language_model.layers.31.mlp.down_proj` has trainable adapter parameters: A `[8,12288]`, B `[4096,8]`, bias `[4096]`, totaling **135,168 FP32 trainable parameters**. The pretrained base remains BF16 and frozen. Rank=8, alpha=8, scaling=1, dropout=0, bias=`none`, `lora_bias=True`; AdamW lr=0.02, weight_decay=0.0, seed=20261024. Exact text inputs, token IDs, gradient norms and training losses are retained. The SFT losses were 1.4448724985, 1.1305598021, 14.2342681885; this short training fixture does not claim convergence.

Two held-out paragraphs exercise complete-model forward calls with labels and `use_cache=False`, eager attention, no quantization and no configuration/layer reduction. The source adapter checkpoint SHA256 is `389428e2e5d9e9e0cb91542a1397984a468def91f04a39ed007e838c2140237c`. All numerical comparisons reload this exact checkpoint. A standard LoRA adapter initialization was used before SFT; the pretrained model was never randomly initialized.

For a single source adapter with combination weight 1 and scaling 1, the intended computation is `base(x) + B(A(x)) + b`. Weighted composition must retain A, B and b exactly. This is an analytical identity, independently of the quality or magnitude of the learned adapter. Full-model reference NLL is **5.032561350294436 nat/token**. Preserving that output is the correctness criterion. Causal localization checks identical A/B factors and identical native inputs to the affected layer; only the bias differs in failing candidates.

## Source and cause

Installed PEFT 0.20.0 `peft/tuners/lora/model.py` exactly matches upstream release source SHA256 `bce383b627ccc546d8298569094fb62cd25154354881938ada038c2cc6bf403f` at release commit `a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e`. Upstream issue/fix provenance is PR #3844, recorded merged on 2026-10-20 with merge commit `1c1ad3bc16655aad3dd395cc1a5d23a5f73c3600`; the exact response, upstream patch and release tag object are in `upstream.json`. This is a known upstream defect reproduced in the declared older runtime, with a local backport and an additional dtype correction. Current upstream execution and novelty are not claimed. No SVD path is used, so this is distinct from the existing SVD double-scaling case.

The original `LoraModel.add_weighted_adapter` combines A/B matrices but omits the learned `lora_B.bias`. Raw composition also leaves a BF16 new bias beside FP32 copied weights, causing a native dtype error. The common measurement configuration calls PEFT's native `_cast_adapter_dtype(identity, True)` on both sides, making the original numerical omission measurable.

The first bias-only backport still fails: `zeros_like(new_adapter.bias)` accumulates into BF16 and rounds a learned FP32 bias before the later cast. Observed bias error is **0.00012194737792015076**. The final candidate constructs the accumulator using `zeros_like(new_adapter.weight[:, 0])`, then computes `sum(weight * source_scaling * source_bias)`. The final bias is already FP32 before the common cast, which is consequently a no-op. Final patch: `repair-fp32.patch`; tested source: `candidate_model_fp32.py`, SHA256 `2751e30a547045150a9cac4036fa7435b04556a062433bbba53200f446de67ab`. Exactly one method changes; shared installation files are untouched. The runner compiles this method from the isolated source and replaces it in process.

## Full-model measurements

All 19,865,600 logits coordinates (80 positions x 248,320 vocabulary entries) and all 78 shifted target-token logprobs are covered. Precision-compare `compare_arrays` and `reduce_scores` supply the metrics.

| Metric | Unpatched | Bias-only backport, FAILED | Final FP32 backport, PASSED |
|---|---:|---:|---:|
| Combined NLL, nat/token | 4.9542036083712 | 5.03333822730033 | 5.03256135029444 |
| NLL absolute deviation | 0.0783577419232389 | 0.000776877005895571 | 0 |
| Target logprob MAE | 0.0836543176496306 | 0.00133983958034943 | 0 |
| Target logprob P99 | 0.29342043876648 | 0.0309062242507937 | 0 |
| Target logprob maximum error | 0.300749778747559 | 0.0820651054382324 | 0 |
| Full logits maximum error | 0.3125 | 0.125 | 0 |
| Full logits RMSE | 0.0504354893383365 | 0.00208648179755255 | 0 |
| Bias maximum error | 0.0600972250103951 | 0.000121947377920151 | 0 |
| Affected layer output maximum error | 0.25 | 0.0625 | 0 |

`test_regression.py` fails on the unpatched workload and the bias-only candidate, and passes on the final native complete-model workload. It checks learned nonzero bias, exact A/B identity, exact bias identity, exact layer input identity, and layer/full-logits errors <=1e-6. Final errors are zero and the array comparator reports exact equality. A separate artifact-only comparison proves the reference logits from the before and final jobs are exactly equal across all 19,865,600 coordinates. Adapter checkpoint SHA256 and token inventories also match. No GPU/model execution was needed for this final artifact comparison.

## Execution evidence and limits

Runtime: PyTorch 2.9.0+cu126, Torch commit `0fabc3ba44823f257e70ce397d989c8de5e362c1`, Transformers 5.3.0, PEFT 0.20.0, Python 3.12.13, one NVIDIA H20Z per job. Final peak allocated CUDA memory: 19015009280 bytes (17.709107 GiB). Each job's native command, actual device/UUID, source identities, terminal status, scheduler identity and cleanup receipt are retained.

The user extended the cap to six native jobs and 60 minutes. Five native jobs were used, each limited to 900 seconds and actually taking 24.15–25.27 seconds. The sixth job is unused. The five trials were: collector metadata failure; real SFT followed by original mixed-dtype failure; unpatched numerical regression failure; bias-only candidate regression failure; final candidate success. A failed run named `bias-green` is an unsuccessful candidate, not a validated repair. All historical run bytes and paths remain unchanged.

Remote runs:

- Job 1 (before, exit 1, 24.402s): `/home/jovyan/projects/precision-skills/projects/precision-repair/cases/pretrained-adapter-repair-20261024T141838Z/runs/gpu-bias-before-20261024T142452Z/native/gpu-bias-before-20261024T142508Z`
- Job 2 (reproduce, exit 1, 24.151s): `/home/jovyan/projects/precision-skills/projects/precision-repair/cases/pretrained-adapter-repair-20261024T141838Z/runs/gpu-bias-reproduce-20261024T142617Z/native/gpu-bias-reproduce-20261024T142628Z`
- Job 3 (red, exit 1, 25.003s): `/home/jovyan/projects/precision-skills/projects/precision-repair/cases/pretrained-adapter-repair-20261024T141838Z/runs/gpu-bias-red-20261024T142856Z/native/gpu-bias-red-20261024T142908Z`
- Job 4 (green, exit 1, 24.153s): `/home/jovyan/projects/precision-skills/projects/precision-repair/cases/pretrained-adapter-repair-20261024T141838Z/runs/gpu-bias-green-20261024T143020Z/native/gpu-bias-green-20261024T143034Z`
- Job 5 (fp32, exit 0, 25.266s): `/home/jovyan/projects/precision-skills/projects/precision-repair/cases/pretrained-adapter-repair-20261024T141838Z/runs/gpu-bias-fp32-20261024T143422Z/native/gpu-bias-fp32-20261024T143435Z`

Local compact evidence:

- Before: `evidence/runs/gpu-bias-red-20261024T142856Z/native/gpu-bias-red-20261024T142908Z/metrics.json`, `regression.json`, `composition-dtypes.json`.
- Failed bias-only candidate: `evidence/runs/gpu-bias-green-20261024T143020Z/native/gpu-bias-green-20261024T143034Z/metrics.json`, `regression.json`, `composition-dtypes.json`.
- Final: `evidence-fp32/native/gpu-bias-fp32-20261024T143435Z/metrics.json`, `regression.json`, `identity.json`, `model-loading.json`, `training.json`, `composition-dtypes.json`.
- Cross-job reference identity: `evidence-reference/reference-identity.json`.
- Verified transfers: `evidence/EXPORT.json`, `evidence-fp32/EXPORT.json`, `evidence-reference/EXPORT.json` (373,433 payload bytes; 115,886 wire bytes).
- Exact launch commands: `launch-*.json`; terminal launcher responses: `launch-*-result.json`. Replays require new timestamped runs.

Full logits, captured layer inputs/outputs, gradients and the trained adapter remain under the remote runs above. No shared installed library, skill, public test or site files were edited. `resources.json` and `execution.json` were left unchanged. `FILES.json` enumerates every local case file and distinguishes those two provided files.

## Reusable improvements actually encountered

1. Capture adapter parameter dtypes at composition boundaries, including the new bias before any autocast. Initialize accumulators from the actual composed weight dtype; a later cast cannot recover values already rounded into BF16. This is supported by the failed bias-only candidate and successful final workload.
2. Collector runtime metadata should treat `hf_device_map` as optional and record actual parameter devices. A mandatory access caused the first instrumentation-only failure after the full model had loaded.
3. No defect was established in shared skill/tool implementations. The launcher retained both infrastructure and regression failures correctly, and comparison/fetch tools returned useful metrics, coverage and integrity receipts. These are proposed workflow/adapter improvements; shared tools were not modified.

Validated scope is unmerged single-source LoRA identity at scaling 1 on two text passages. Visual inputs, generation/cache, multi-source composition, SVD, quantization and distributed training remain untested.
