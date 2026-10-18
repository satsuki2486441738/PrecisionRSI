---
name: precision-skill
description: Collect and compare numerical results across hardware and software execution stacks for inference, SFT, LoRA, and RL. Use for fixed-input scoring, loss, gradient, or parameter-update comparisons across execution stacks.
---

# Precision comparison

Select the execution setups to compare from the task. Each side has an ID; hardware, runtime and framework are its configuration. Choose the task path and read only its references.

- **Existing captures:** use [tools](references/tools.md) directly. Check model/input/state identities and the requested stage/coordinates.
- **Collection needed:** read [preparation](references/collection.md), then [inference](references/inference.md) or [training](references/training.md). Reuse an existing collector when available.

Read integration details when creating or adapting a collector, selecting the relevant section:

- [vLLM / SGLang](references/engines.md): serving scores and cached decode.
- [TRL / verl](references/rl-frameworks.md): native RL scoring and loss.
- [Training backends](references/training-backends.md): distributed tensors and optimizer capture.
- [Other/custom](references/custom-integration.md): frameworks, backends or environments whose required interfaces are not covered above.

Read [metric definitions](references/comparison.md) for formula explanations or custom calculations. Return settings used, principal differences, compared coordinate counts, largest discrepancies and output paths. Evaluate supplied tolerances against the computed metrics.
