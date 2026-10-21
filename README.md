# Precision-Skills: Numerical Precision Comparison Across Devices and Environments

English | [简体中文](README.zh-CN.md)

Agent Skills for numerical precision comparison and repair across execution environments.

## Use

Choose the skill for the requested task and copy its directory into your client's skill directory:

| Skill | Task and result |
|---|---|
| [Precision Compare](skills/precision-skill/SKILL.md) | Collect and compare numerical results; return metrics, coverage and evidence |
| [Precision Repair](skills/precision-repair/SKILL.md) | Establish the cause of a numerical defect, patch the implementation and verify the original workload |

Install both to reuse the bundled collection and comparison tools during repair. Precision Repair can also work from a failing native test using task-owned tooling.

For comparison, provide the task and existing artifacts, or the model, data and platform connections needed for collection. The agent prepares the environment, integrates native execution, writes missing collection/comparison code and runs the task. Reusable tools are extended during skill development.

Examples:

- Compare the gradients and stored updates in these two execution manifests.
- Collect a LoRA SFT step on two execution stacks using the same batch and initial adapter state, then compare it.

[SKILL.md](skills/precision-skill/SKILL.md) routes collection and comparison. The three task recipes in `references/` cover inference, SFT (full-parameter / LoRA) and RL: what to capture, where to integrate with native frameworks (including Other/custom), and which metrics to report. `comparison.md` defines coordinate pairing, common metrics and tool usage; distribution metrics are defined in the inference recipe. Training capture stages are defined in SFT and reused by RL.

Results contain actual settings, metric values and aggregations, coverage counts, maximum-error coordinates and artifact paths. The task determines common conditions and allowed per-side differences across peer execution setups. Comparison metrics are symmetric. Collection retains native operators and records each side's dependencies.

For internal layer comparison, the agent first confirms scope and resource limits, then autonomously integrates collectors and refines sampling. The [layerwise guide](skills/precision-skill/references/layerwise.md) includes msProbe capabilities, integration and trial checks, with native hooks for other execution paths.

## Typical cases

### Ascend 910C (昇腾) NPU and NVIDIA H200 GPU

Paired collection and numerical comparison with Qwen3.5-9B-Base, using each platform's native software stack:

| Framework | Completed comparison |
|---|---|
| PyTorch / Transformers; PEFT for LoRA | Full-forward inference, full-parameter SFT and LoRA SFT |
| Transformers Trainer | Native full-parameter SFT single step; target scores and sampled gradients/stored updates |
| vLLM / vLLM-Ascend | Native prefill target scores; fixed-history cached inference distributions and target scores |
| SGLang | Native cached inference with fixed continuations; prompt scores and preconstraint distributions |
| TRL | Fixed-trajectory GRPO action scores, loss, gradients and one-step parameter updates |
| verl / FSDP2 | Fixed-trajectory GRPO action scores, loss, gradients and one-step parameter updates |

## Tools and development

The skill organizes the task; task adapters integrate native execution; the [tool library](skills/precision-skill/references/comparison.md#tools) provides reusable operations. `scripts/` contains CLI tools, callable Python tools and their supporting modules. Tool docstrings define inputs and outputs; task recipes select capture boundaries and metrics.

Tools check runtime initialization and imported interfaces, capture fixed-history inference distributions, plan internal capture, export and validate artifacts, and compare distributions, tensor observations (including activations), gradients, updates and target scores. msProbe is an optional collection backend; the bundled planner generates bounded L0 selections and estimates tensor payload, while task adapters execute collection. Add missing task integration as needed and extract reusable operations into the tool library during development. Array tools require NumPy; target-score tools use the standard library.

[Evidence tools](skills/precision-skill/references/comparison.md#portable-evidence) package selected artifacts with source/export hashes and replay retained comparisons through installed skill CLIs.

For a concrete collection task and the expected agent workflow, see the [fixed-history inference example](skills/precision-skill/examples/fixed-history-inference.md), from environment preparation through capture, validation and reporting.

Compare FP32 and BF16 gradients for the same small classifier and batch with the [runnable gradient example](skills/precision-skill/examples/compare_gradients.py). Run from the skill directory; requires Torch and NumPy:

```sh
python3 examples/compare_gradients.py
```

It runs both forward/backward passes on CPU, then prints losses, gradient-difference metrics and the generated artifact directory.

Repository layout and test commands are in [AGENTS.md](AGENTS.md).
