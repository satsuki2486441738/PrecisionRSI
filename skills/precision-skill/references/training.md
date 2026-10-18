# Training collection

Read the common state below and select [SFT/LoRA](sft.md) or [RL](rl.md).

## Common training state

Replay identical logical parameters, optimizer state and microbatch order. Match accumulation, scaling, communication, clipping and actual random inputs.

Collect the requested stages:

| Stage | Observations |
|---|---|
| Objective | Scores, masks, native losses, coefficients and effective denominators |
| Backward | All present gradients after accumulation, unscale and synchronization; presence inventory |
| Clipping | Native norm/coefficient and postclip gradients when enabled |
| Update | Stored before/after parameters, including unchanged coordinates; applied/skipped event |

Preserve native dtypes and logical coordinates. Use [tools](tools.md) for comparison. Read [backend capture](training-backends.md) when implementing distributed collection.

## Reusing a training baseline

When establishing a stateful-optimizer baseline, collect cold and common-restored steps, including parameters, moments and step counter. Reuse accepted baselines for unchanged paths; repeat affected stages after optimizer, precision, objective, sharding or dependency changes.
