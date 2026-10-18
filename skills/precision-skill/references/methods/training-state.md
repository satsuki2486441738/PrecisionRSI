# Training state and capture

## Stages

Replay identical logical parameters, optimizer state and microbatch order. Match accumulation, scaling, communication, clipping and actual random inputs.

Collect the requested stages:

| Stage | Observations |
|---|---|
| Objective | Scores, masks, native losses, coefficients and effective denominators |
| Backward | All present gradients after accumulation, unscale and synchronization; presence inventory |
| Clipping | Native norm/coefficient and postclip gradients when enabled |
| Update | Stored before/after parameters, including unchanged coordinates; applied/skipped event |

Preserve native dtypes and logical coordinates. Use [tools](tools.md) for comparison.

## Distributed capture

| Backend | Resolve |
|---|---|
| FSDP / ZeRO | Parameter/gradient sharding, synchronization, offload and optimizer ownership |
| Megatron-Core / MindSpeed | Engine/bridge, TP/PP/CP/DP, applicable EP/sequence parallelism and pipeline stages |

Resolve `mbridge` versus `megatron.bridge` and the NPU MindSpeed adaptation. For hybrid attention, identify convolution, normalization and recurrence implementations.

- Capture synchronized gradients from native storage, such as Megatron `main_grad`, after accumulation and unscale.
- Capture postclip gradients at the native clipping boundary before optimizer mutation.
- Capture stored parameters after optimizer mutation and model copy-back. Record master/state dtypes separately.

Map fused projections, shards, vocabulary padding, tied weights and replicas into logical parameters. Verify converted initial values and complete coordinates. Retain reduction tags; deduplicate equal replicas and keep differing replicas as explicit views. Adapt arrays to the [tool interfaces](tools.md) with native dtypes preserved.

## Reusing training captures

When establishing a stateful-optimizer baseline, collect cold and common-restored steps, including parameters, moments and step counter. Reuse accepted baselines for unchanged paths; repeat affected stages after optimizer, precision, objective, sharding or dependency changes.
