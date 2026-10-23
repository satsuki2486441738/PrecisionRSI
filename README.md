# Precision-Skills: Numerical Precision Comparison and Repair

English | [简体中文](README.zh-CN.md)

Agent Skills for framework adaptation, numerical precision comparison and repair across devices and execution environments.

## Use

Choose the skill for the requested task and copy its directory into your client's skill directory:

| Skill | Task and result |
|---|---|
| [Framework Adapt](skills/framework-adapt/SKILL.md) | Deploy or adapt a framework for the target model, hardware and workload; deliver reproducible execution and validation evidence |
| [Precision Compare](skills/precision-compare/SKILL.md) | Collect and compare numerical results; return metrics, coverage and evidence |
| [Precision Repair](skills/precision-repair/SKILL.md) | Establish the cause of a numerical defect, patch the implementation and verify the original workload |
| [Operator Optimize](skills/operator-optimize/SKILL.md) | Placeholder; not implemented |

For end-to-end workloads, install the three implemented packages side by side. Individual tasks can use the selected component; adaptation and repair reuse tools from `precision-compare`. `precision-compare` is the renamed comparison package previously called `precision-skill`; replace the old installed directory when updating.

For an end-to-end task, provide the model workload, target execution stacks and resource scope. The agent establishes missing native execution, compares the requested observations and repairs confirmed problems when requested. The three skills share a [workflow and resource catalog](skills/precision-compare/references/comparison.md#shared-tasks): valid preparation and captures are reused, failures identify the next component, and code changes rerun affected verification. Each component also works independently.

For comparison, provide the task and existing artifacts, or the model, data and platform connections needed for collection. The agent prepares the environment, integrates native execution, writes missing collection/comparison code and runs the task. Reusable tools are extended during skill development.

Examples:

- Compare the gradients and stored updates in these two execution manifests.
- Run this model's SFT workload on the supplied execution stacks, compare numerical results, fix confirmed correctness problems and verify the original workload.

[SKILL.md](skills/precision-compare/SKILL.md) routes collection and comparison. The three task recipes in `references/` cover inference, SFT (full-parameter / LoRA) and RL: what to capture, where to integrate with native frameworks (including Other/custom), and which metrics to report. `comparison.md` defines coordinate pairing, common metrics and tool usage; distribution metrics are defined in the inference recipe. Training capture stages are defined in SFT and reused by RL.

Existing corresponding arrays can use [compare_arrays](skills/precision-compare/scripts/array_compare.py) directly, with exact equality, coverage and bounded findings; error metrics are optional. Read collection recipes when new capture is needed, and use artifact interfaces for required mapping or execution validation.

Results contain the requested comparison results, actual settings, coverage counts, difference coordinates and reproduction artifacts. The task determines common conditions and allowed per-side differences across peer execution setups. Comparison metrics are symmetric. Collection retains native operators and records each side's dependencies.

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

The skills organize one workload; task adapters integrate native execution; the [tool library](skills/precision-compare/references/comparison.md#tools) executes and reuses its stages. `scripts/` contains CLI tools, callable Python tools and their supporting modules. Tool docstrings define inputs and outputs; `tool_help.py` queries selected interfaces without importing dependencies. Task recipes select capture boundaries and metrics.

Tools check runtime initialization and imported interfaces, record command runs, capture fixed-history inference distributions, plan internal capture, export and validate artifacts, and compare distributions, tensor observations (including activations), gradients, updates and target scores. Tensor diagnostics also inventory NaN/Inf locations and report finite-subset coverage and metrics separately. msProbe is an optional collection backend; the bundled planner generates bounded L0 selections and estimates tensor payload, while task adapters execute collection. Add missing task integration as needed and extract reusable operations into the tool library during development. Nested-state comparison reuses array arithmetic while checking checkpoint structure and metadata. Array tools require NumPy; target-score tools use the standard library.

[Evidence tools](skills/precision-compare/references/comparison.md#portable-evidence) package selected artifacts with source/export hashes and replay retained comparisons through installed skill CLIs.

For a concrete collection task and the expected agent workflow, see the [fixed-history inference example](skills/precision-compare/examples/fixed-history-inference.md), from environment preparation through capture, validation and reporting.

Compare FP32 and BF16 gradients for the same small classifier and batch with the [runnable gradient example](skills/precision-compare/examples/compare_gradients.py). Run from the skill directory; requires Torch and NumPy:

```sh
python3 examples/compare_gradients.py
```

It runs both forward/backward passes on CPU and directly compares named gradient arrays. The run retains two NPZ captures, actual configurations and a comparison report, then prints losses, metrics and the artifact directory.

Run the [native end-to-end example](skills/precision-compare/examples/workflow_demo.py) from the skill directory:

```sh
python3 examples/workflow_demo.py --output-root /tmp/precision-workloads
```

It runs a two-step Torch SFT fixture, detects a controlled objective defect, applies its known correction and revalidates the original trajectory while reusing preparation. The final replay launches zero native commands.

Repository layout and test commands are in [AGENTS.md](AGENTS.md).

## Evaluation and repair history

The [history website](https://satsuki2486441738.github.io/precision-skills/) has two tabs: **Evaluations** shows inference/SFT/RL comparison settings, numerical metrics, coverage and evidence; **Repairs** retains library-grouped fixes, causes and before/after verification. Records and public evidence are maintained in [site/](site/README.md).
