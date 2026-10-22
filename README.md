# Precision-Skills: Numerical Comparison, Repair and Operator Optimization

English | [简体中文](README.zh-CN.md)

Three Agent Skills for numerical precision comparison, repair and operator optimization across devices and execution environments.

## Use

Choose the skill for the requested task and copy its directory into your client's skill directory:

| Skill | Task and result |
|---|---|
| [Precision Compare](skills/precision-compare/SKILL.md) | Collect and compare numerical results; return metrics, coverage and evidence |
| [Precision Repair](skills/precision-repair/SKILL.md) | Establish the cause of a numerical defect, patch the implementation and verify the original workload |
| [Operator Optimize](skills/operator-optimize/SKILL.md) | Improve operator numerical accuracy, latency or memory and verify the original workload |

Install the three directories side by side to use the linked workflow and shared comparison tools. `precision-compare` is the renamed comparison package previously called `precision-skill`; replace the old installed directory when updating.

Start with comparison to collect metrics and coverage. When repair is requested, carry the contract, per-side manifests/configurations, original inputs/state, source identities, native commands and reports into repair. Establish correctness evidence to identify the responsible behavior, then patch and rerun the original comparison. For an operator accuracy or efficiency objective, use operator optimization with the same workload and evidence, adding latency and memory measurements. Each skill can also start from an existing workload or test; collection is reused when its inputs and conditions still match.

For comparison, provide the task and existing artifacts, or the model, data and platform connections needed for collection. The agent prepares the environment, integrates native execution, writes missing collection/comparison code and runs the task. Reusable tools are extended during skill development.

Examples:

- Compare the gradients and stored updates in these two execution manifests.
- Collect a LoRA SFT step on two execution stacks using the same batch and initial adapter state, then compare it.

[SKILL.md](skills/precision-compare/SKILL.md) routes collection and comparison. The three task recipes in `references/` cover inference, SFT (full-parameter / LoRA) and RL: what to capture, where to integrate with native frameworks (including Other/custom), and which metrics to report. `comparison.md` defines coordinate pairing, common metrics and tool usage; distribution metrics are defined in the inference recipe. Training capture stages are defined in SFT and reused by RL.

Results contain actual settings, metric values and aggregations, coverage counts, maximum-error coordinates and artifact paths. The task determines common conditions and allowed per-side differences across peer execution setups. Comparison metrics are symmetric. Collection retains native operators and records each side's dependencies.

For internal layer comparison, the agent first confirms scope and resource limits, then autonomously integrates collectors and refines sampling. The [layerwise guide](skills/precision-compare/references/layerwise.md) includes msProbe capabilities, integration and trial checks, with native hooks for other execution paths.

## Typical cases

### Ascend 910C (昇腾) NPU and NVIDIA H200 GPU

Paired collection and numerical comparison with Qwen3.5-9B-Base, using each platform's native software stack:

| Framework | Completed comparison |
|---|---|
| PyTorch / Transformers; PEFT for LoRA | Full-forward inference, full-parameter SFT and LoRA SFT |
| Transformers Trainer | Native full-parameter SFT single step (sampled gradients/updates); Single-device and two-device FSDP2 LoRA SFT with gradient accumulation, two AdamW steps and checkpoint restoration (complete adapter gradients, updates and optimizer state); target scores |
| vLLM / vLLM-Ascend | Native prefill target scores; fixed-history cached inference distributions and target scores |
| SGLang | Native cached inference with fixed continuations; prompt scores and preconstraint distributions |
| TRL | Native SFTTrainer LoRA SFT with completion-only labels, gradient accumulation, two AdamW steps and native checkpoint restoration (complete adapter gradients, updates and optimizer state); fixed-trajectory GRPO action scores, loss, gradients and one-step parameter updates |
| verl / FSDP2 | Fixed-trajectory GRPO action scores, loss, gradients and one-step parameter updates |

### Precision repair

Qwen3.5-9B-Base: repaired zero-centered RMSNorm parameter semantics in a pinned historical stack. Verification covered native NPU operator forward/backward, checkpoint round trips, and the original TP4 model forward with unchanged loaded parameter bytes.

Additional historical backports were verified using small native CPU workloads:

| Library / pinned version | Repair | Verification |
|---|---|---|
| TRL 0.7.10 | IPO completion logprob averaging | Native DPOTrainer loss, all gradients and one SGD update; 6 regressions |
| PEFT 0.17.1 | Duplicate LoRA scaling in SVD adapter combination | Forward, input gradients, checkpoint reload and merge; 24 workloads |
| Transformers 4.51.3 | Qwen2 attention mask ignores disabled sliding window | Eager/SDPA full forward and dynamic/static cache; 18 workloads and 16 mask cases |

The [repair history](https://satsuki2486441738.github.io/precision-skills/) records affected and upstream-fixed revisions, patches, before/after metrics and verification scope.


## Tools and development

The skill organizes the task; task adapters integrate native execution; the [tool library](skills/precision-compare/references/comparison.md#tools) provides reusable operations. `scripts/` contains CLI tools, callable Python tools and their supporting modules. Tool docstrings define inputs and outputs; task recipes select capture boundaries and metrics.

Tools check runtime initialization and imported interfaces, record command runs, capture fixed-history inference distributions, plan internal capture, export and validate artifacts, and compare distributions, tensor observations (including activations), gradients, updates and target scores. msProbe is an optional collection backend; the bundled planner generates bounded L0 selections and estimates tensor payload, while task adapters execute collection. Add missing task integration as needed and extract reusable operations into the tool library during development. Array tools require NumPy; target-score tools use the standard library.

[Evidence tools](skills/precision-compare/references/comparison.md#portable-evidence) package selected artifacts with source/export hashes and replay retained comparisons through installed skill CLIs.

For a concrete collection task and the expected agent workflow, see the [fixed-history inference example](skills/precision-compare/examples/fixed-history-inference.md), from environment preparation through capture, validation and reporting.

Compare FP32 and BF16 gradients for the same small classifier and batch with the [runnable gradient example](skills/precision-compare/examples/compare_gradients.py). Run from the skill directory; requires Torch and NumPy:

```sh
python3 examples/compare_gradients.py
```

It runs both forward/backward passes on CPU, then prints losses, gradient-difference metrics and the generated artifact directory.

Repository layout and test commands are in [AGENTS.md](AGENTS.md).

## Evaluation and repair history

The [history website](https://satsuki2486441738.github.io/precision-skills/) has two tabs: **Evaluations** shows inference/SFT/RL comparison settings, numerical metrics, coverage and evidence; **Repairs** retains library-grouped fixes, causes and before/after verification. Records and public evidence are maintained in [site/](site/README.md).
