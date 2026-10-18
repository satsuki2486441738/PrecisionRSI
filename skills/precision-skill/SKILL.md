---
name: precision-skill
description: Collect and compare numerical results across hardware and software execution stacks for inference, SFT, LoRA, and RL. Use for fixed-input scoring, loss, gradient, or parameter-update comparisons across execution stacks.
---

# Precision comparison

Select the execution setups to compare from the task. Each side has an ID; hardware, runtime and framework are its configuration.

- **Existing captures:** use [tools](references/methods/tools.md). Check model/input/state identities and the requested stage/coordinates.
- **Collection needed:** read [collection](references/methods/collection.md), then the task recipe below. Reuse an existing collector when available.

| Task | Recipe |
|---|---|
| Inference | [Scores, cache stages and serving engines](references/tasks/inference.md) |
| SFT (full / LoRA) | [Training targets, model and adapter state](references/tasks/sft.md) |
| RL | [Fixed trajectories and framework integration](references/tasks/rl.md) |

Each task recipe defines capture requirements, framework integration (including Other/custom) and the metrics to report. Read the linked methods for training state, [native adaptation](references/methods/collection.md#adapt-an-unfamiliar-implementation) and [metric formulas](references/methods/metrics.md) as needed.

Return settings used, principal differences, compared coordinate counts, largest discrepancies and output paths. Evaluate supplied tolerances against the computed metrics.
