# TRL and verl

Apply the [RL protocol](training.md#rl) to each framework's GPU/NPU pair. Inspect the installed trainer, loss variant and backend; hooks and supported algorithms vary by release.

## Update boundary

Replay the fixed trajectory/dependencies into native loss, backward and optimizer execution. Identify one actual optimizer step per active model, its accumulation window and actor/critic order; a rollout batch or epoch may contain several steps. Scheduled inactive models remain outside that step's coverage.

Use backend-aware gradient/parameter access for FSDP/ZeRO/Megatron; callbacks may not expose synchronized full gradients. Follow [training collection](training.md#common-training-state) for stages and coverage. Record actual integration scope, such as a loss adapter versus the complete trainer loop.

## TRL

Trace batch preparation, action scoring, loss and optimizer integration. Capture effective loss inputs, including completion/tool/truncation masks, group normalization and rollout-correction weights. Preserve dynamic masks as intermediates; numerical branch changes are not missing trajectory coordinates.

Old scores may be a retained snapshot or detached current scores. Preserve that rule and detach semantics rather than substituting a common denominator. Verify temperature and other score transforms used by the trainer; raw inference scores may define a different objective. Observe the selected native or fused loss path.

## verl

Trace rollout, reward/reference/value computation, advantages and actor/critic update workers. Retain sample/group/action mappings through repeats, filtering, balancing, packing, padding removal and rank redistribution.

Distinguish rollout scores from a recomputed old-policy anchor; some configurations deliberately reuse rollout scores. Preserve correction/rejection weights, reward/KL placement, estimator and loss aggregation. Capture actual training-backend observations; RPC averages cannot supply full gradients or step objectives.

For vLLM/SGLang rollout diagnosis, read [engines](engines.md). Fixed-trajectory trainer comparison does not require regenerating rollout or adding inference metrics.


## Megatron-Core and MindSpeed

For a verl Megatron backend pair, GPU normally uses Megatron-Core with CUDA libraries; NPU uses Megatron-Core with MindSpeed adaptation. MindSpeed Core, MindSpeed-LLM and MindSpeed-RL are distinct projects. Record the actual engine class and patched modules rather than inferring the execution path from a backend label or an installed package.

### Shared implementation gate

Pin the same verl and Megatron-Core source revisions on both sides. Also align the model bridge: `mbridge` and NVIDIA Megatron Bridge (`megatron.bridge`) are different implementations, even when both can load the same Hugging Face checkpoint. Inspect effective `use_mbridge` / `vanilla_mbridge` settings and model registration. A framework example with different per-device versions or bridges is a starting point for investigation, not an aligned comparison contract.

Check the selected bridge's declared dependency constraints against the common environment before installation. Separate an upstream requirement conflict from an import failure and from a device execution failure. Use isolated environments; do not silently override dependency bounds, alter shared source on only one side, or substitute a different model to obtain a run. If shared versions cannot be aligned, retain the evidence and confirm the changed approach with the user.

Verify support for the exact architecture and variant, including hybrid attention, model configuration, weight conversion and backward. A working related MoE example does not establish dense-model coverage. For hybrid linear attention, record the actual convolution, normalization and recurrence implementations and any fallback; a deterministic option may select a different implementation. Preserve the native CUDA/CANN paths within the agreed numerical policy.

### Native step observations

Keep the fixed-trajectory RL protocol and existing metrics. Record TP/PP/CP/DP and, where applicable, EP, sequence parallelism, packing and virtual pipeline stages. Track action coordinates through sequence redistribution; compare only after mapping them back to the common trajectory.

Capture synchronized native gradients from their actual storage (for example, Megatron `main_grad` buffers), before and after native clipping. Mixed-precision optimizers may maintain FP32 master parameters separately from BF16 model parameters. Record each dtype, the optimizer state, whether a step was skipped and when master parameters were copied back. Compare actual stored model updates, and identify master-parameter observations separately when needed for localization.

Export tensors into a common logical parameter coordinate system. Preserve the mapping for fused QKV/gate projections, tensor/pipeline shards, vocabulary padding, tied parameters and replicated parameters. Validate complete non-overlapping coverage after excluding declared padding and deduplicating replicas; matching rank-local filenames or element counts is insufficient. Verify converted initial weights before attributing subsequent differences to native operators.

Keep backend-specific collectors and conversion maps in the case. A completed FSDP comparison does not validate this backend; source inspection, import checks, device execution and numerical agreement are separate evidence.
