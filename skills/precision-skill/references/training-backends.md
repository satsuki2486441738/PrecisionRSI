# Distributed capture hooks

Use these hooks when implementing the [training stages](training.md#common-training-state).

| Backend | Resolve |
|---|---|
| FSDP / ZeRO | Parameter/gradient sharding, synchronization, offload and optimizer ownership |
| Megatron-Core / MindSpeed | Engine/bridge, TP/PP/CP/DP, applicable EP/sequence parallelism and pipeline stages |

Resolve `mbridge` versus `megatron.bridge` and the NPU MindSpeed adaptation. For hybrid attention, identify convolution, normalization and recurrence implementations.

- Capture synchronized gradients from native storage, such as Megatron `main_grad`, after accumulation and unscale.
- Capture postclip gradients at the native clipping boundary before optimizer mutation.
- Capture stored parameters after optimizer mutation and model copy-back. Record master/state dtypes separately.

Map fused projections, shards, vocabulary padding, tied weights and replicas into logical parameters. Verify converted initial values and complete coordinates. Retain reduction tags; deduplicate equal replicas and keep differing replicas as explicit views. Adapt arrays to the [tool interfaces](tools.md) with native dtypes preserved.
