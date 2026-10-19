# precision-skill

English | [简体中文](README.zh-CN.md)

An Agent Skill for inference, SFT, LoRA and RL numerical comparison across hardware and software execution stacks.

## Use

Copy [skills/precision-skill](skills/precision-skill/) into your client's skill directory. Provide the task and existing artifacts, or the model, data and platform connections needed for collection. The agent prepares the environment, integrates native execution, writes missing collection/comparison code and runs the task. Reusable tools are extended during skill development.

Examples:

- Compare the gradients and stored updates in these two execution manifests.
- Collect a LoRA SFT step on two execution stacks using the same batch and initial adapter state, then compare it.

[SKILL.md](skills/precision-skill/SKILL.md) routes collection and comparison. The three task recipes in `references/` cover inference, SFT (full-parameter / LoRA) and RL: what to capture, where to integrate with native frameworks (including Other/custom), and which metrics to report. `comparison.md` defines coordinate pairing, common metrics and tool usage; distribution metrics are defined in the inference recipe. Training capture stages are defined in SFT and reused by RL.

Results contain actual settings, metric values and aggregations, coverage counts, maximum-error coordinates and artifact paths. The task determines common conditions and allowed per-side differences across peer execution setups. Comparison metrics are symmetric. Collection retains native operators and records each side's dependencies.

## Typical cases

### Ascend 910C (昇腾) NPU and NVIDIA H200 GPU

Paired collection and numerical comparison with Qwen3.5-9B-Base, using each platform's native software stack:

| Framework | Completed comparison |
|---|---|
| PyTorch / Transformers; PEFT for LoRA | Full-forward inference, full-parameter SFT and LoRA SFT |
| TRL | Fixed-trajectory GRPO action scores, loss, gradients and one-step parameter updates |
| verl / FSDP2 | Fixed-trajectory GRPO action scores, loss, gradients and one-step parameter updates |

## Tools and development

[Offline tools](skills/precision-skill/references/comparison.md) compare distributions, gradients, updates and target scores. Array tools require NumPy; target-score tools use the standard library. Shared helpers support per-side execution conditions, cross-host distribution queries, logical tensor exports and explicit replica views.

Run the [synthetic artifact example](skills/precision-skill/examples/tensors.py) from the skill directory:

```sh
python3 examples/tensors.py
```

It creates a fresh temporary run containing two gradient captures, validation reports and numerical comparison output.

Repository layout and test commands are in [AGENTS.md](AGENTS.md).
