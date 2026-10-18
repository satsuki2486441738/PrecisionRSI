# precision-skill

English | [简体中文](README.zh-CN.md)

An Agent Skill for inference, SFT, LoRA and RL numerical comparison across hardware and software execution stacks.

## Use

Copy [skills/precision-skill](skills/precision-skill/) into your client's skill directory. Provide the task and existing artifacts, or the model, data and platform connections needed for collection.

Examples:

- Compare the gradients and stored updates in these two execution manifests.
- Collect a LoRA SFT step on two execution stacks using the same batch and initial adapter state, then compare it.

[SKILL.md](skills/precision-skill/SKILL.md) selects the path: existing captures go directly to a comparator; collection reads the selected task protocol. Task recipes in `references/tasks/` include framework integration. Collection, training state, metrics and tool usage are in `references/methods/` and are read as needed.

Results contain settings, principal differences, compared coordinates and largest discrepancies. The task selects peer execution setups, which may use the same or different hardware. Comparison metrics are symmetric. Collection retains each setup's native operators and pinned dependencies.

## Tools and development

[Offline tools](skills/precision-skill/references/methods/tools.md) compare distributions, gradients, updates and target scores. Artifact tools require NumPy; the cache-off target-score comparator uses the standard library.

Repository layout and test commands are in [AGENTS.md](AGENTS.md).
