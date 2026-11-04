# Gemma 4 12B: four independent H100 discovery captures

**Execution and artifact verification completed for this scoped pilot.** This
record does not report numerical acceptance or a formal precision comparison.

On 2026-11-05, a four-GPU task ran four separate, concurrently launched
single-GPU inference workers on NVIDIA H100 80GB HBM3 devices. All four workers
exited with code zero, the coordinator reported completion, and the platform
reported the task as successful. Independent checks matched the recorded input,
binding, command and runtime hashes; each worker's recorded GPU identity matched
its actual PyTorch device, and the four devices were distinct.

The workers were assigned the four frozen WikiText-2 discovery examples
`001`–`004`. Each input has 128 prompt tokens including BOS and 32 predetermined
continuation targets. Independent checks confirmed 32/32 captured rows per
input, each containing 262,144 finite float64 log probabilities, and confirmed
the prompt length, cached decode history and target coordinates. These are
English-prose discovery examples, not an independent or representative
validation set. The derived rows and native run artifacts remain private and
are **not publicly replayable** from this record.

The pinned base model was `google/gemma-4-12B` revision
`1a2c7729392586f5ecddae8807ee2ef741eabb9a`. The common execution used BF16,
eager attention, seed 0, PyTorch `2.7.0+cu128`, Transformers `5.10.1` and CUDA
runtime 12.8. The frozen numerical policy specifies float32 matmul precision
`highest`, CUDA matmul TF32 `false`, cuDNN TF32 `true`, deterministic algorithms
and cuDNN deterministic/benchmark `false`, `PYTHONHASHSEED=0`, and an unset
`CUBLAS_WORKSPACE_CONFIG`. Independent checks confirmed the recorded policy
before and after execution matched the frozen policy. The task checked a
verified snapshot receipt and current file sizes; that receipt alone does not
independently prove the bytes loaded by each worker.

An exploratory read-only check compared example `001` with the first earlier
single-H100 run using the same input, model, software and numerical policy. The
runs used distinct H100 devices, and the later run executed alongside three
other workers. All 32 × 262,144 derived log-probability values were elementwise
equal; target log-probability MAE and maximum absolute difference were 0 nat.
The earlier three-repeat admission gate does not apply to the ACP run because
its required preflight receipt is absent; no formal comparison run was created.
This single-example observation is limited to the retained derived rows, not
raw logits or other inputs. Those arrays remain private and are not publicly
replayable from this record.

These are four independent single-GPU model executions over different inputs.
They are **not** tensor parallelism, distributed model inference, a cross-platform
precision comparison, or a four-GPU numerical equivalence result. The earlier
H100–RTX 5090 comparison retains its original single-input scope. No held-out
validation, broad input coverage, multimodal execution, numerical thresholds,
causal attribution or model-wide acceptance is established by this record.
Detailed allocation records remain private.
