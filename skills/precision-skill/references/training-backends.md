# Training backend integration

Apply the [common training protocol](training.md#common-training-state) to the execution backend selected by the trainer. The backend determines tensor storage, synchronization and optimizer boundaries; the task determines scores, objectives and metrics.

## Execution identity

Establish the actual engine class, model bridge and imported source. Use [environment alignment](environment.md#shared-implementation) for independently pinned platform stacks or a requested shared-implementation scope.

| Backend | Native access and alignment |
|---|---|
| FSDP / ZeRO | Identify parameter/gradient sharding, synchronization points, offload and optimizer ownership. Export logical tensors after the step's accumulation and communication complete. |
| Megatron-Core / MindSpeed | Select a compatible Core/model bridge on each side and align effective model semantics. GPU uses the CUDA stack; NPU adds MindSpeed adaptation. Record TP/PP/CP/DP and applicable EP, sequence parallelism and virtual pipeline stages. |

`mbridge` and NVIDIA Megatron Bridge (`megatron.bridge`) are separate implementations. Resolve the effective bridge selection and support for the exact model architecture and variant, including conversion and backward. MindSpeed Core is the Megatron adaptation layer; MindSpeed-LLM and MindSpeed-RL are separate application frameworks.

For hybrid attention, record the effective convolution, normalization and recurrence implementations, including platform fallbacks. Packing, deterministic settings and other configuration choices can change the executed path. Retain these identities with the numerical policy.

## Capture boundaries

Trace one applied or skipped optimizer step through these native stages:

1. Accumulation, unscale and gradient communication complete. Capture synchronized gradients from their actual storage, such as Megatron `main_grad` buffers.
2. Native clipping completes. Capture the norm, coefficient and postclip gradients before optimizer mutation.
3. The optimizer updates its parameters and copies values back to the model. Capture stored model parameters and step status; identify master parameters and their dtype separately when present.

Calling a native finalize/synchronization API establishes a capture boundary, not equality of every TP replica. Retain the relevant parameter reduction tags and check replica consistency separately; report an unresolved replica as a native finding or mapping uncertainty rather than silently averaging it.

Mixed-precision optimizers may update FP32 master parameters while the model stores BF16 values. State and dtype evidence establish what the recorded gradient and update represent. Framework callbacks and RPC reductions are useful only when they expose the required stage and scope.

## Logical coordinates

Export observations into the common trajectory and parameter coordinates before comparison. Retain conversion maps for fused projections, tensor/pipeline shards, vocabulary padding, tied weights and replicated parameters. Verify initial converted weights, then validate complete non-overlapping coverage with declared padding excluded and replicas deduplicated.

Keep native observations and conversion maps with the case; reuse the existing offline comparators after adaptation. Report source preparation, import/conversion checks, device execution and paired numerical results as distinct evidence. A backend baseline covers its recorded implementation and configuration.
