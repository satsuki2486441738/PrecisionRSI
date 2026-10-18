# precision-skill

English | [简体中文](README.zh-CN.md)

An Agent Skill for inference, SFT, LoRA and RL numerical comparison across hardware and software execution stacks.

## Use

Copy [skills/precision-skill](skills/precision-skill/) into your client's skill directory. Provide the task and existing artifacts, or the model, data and platform connections needed for collection.

Examples:

- Compare the gradients and stored updates in these two execution manifests.
- Collect a LoRA SFT step on two execution stacks using the same batch and initial adapter state, then compare it.

[SKILL.md](skills/precision-skill/SKILL.md) routes collection and comparison. The three task recipes in `references/` cover inference, SFT (full-parameter / LoRA) and RL: what to capture, where to integrate with native frameworks (including Other/custom), and which metrics to report. `comparison.md` defines coordinate pairing, common metrics and tool usage; distribution metrics are defined in the inference recipe. Training capture stages are defined in SFT and reused by RL.

Results contain settings, principal differences, compared coordinates and largest discrepancies. The task selects peer execution setups, which may use the same or different hardware. Comparison metrics are symmetric. Collection retains each setup's native operators and pinned dependencies.

## Tools and development

[Offline tools](skills/precision-skill/references/comparison.md) compare distributions, gradients, updates and target scores. Artifact tools require NumPy; the cache-off target-score comparator uses the standard library.

Repository layout and test commands are in [AGENTS.md](AGENTS.md).
