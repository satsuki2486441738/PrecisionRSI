<p align="center">
  <img src="docs/images/numericbridge-logo.svg" width="88" height="88" alt="NumericBridge logo">
</p>

<h1 align="center">NumericBridge</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-009688.svg" alt="License: MIT"></a>
  <a href="https://github.com/satsuki2486441738/NumericBridge/actions/workflows/ci.yml"><img src="https://github.com/satsuki2486441738/NumericBridge/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status"></a>
  <a href="https://satsuki2486441738.github.io/NumericBridge/"><img src="https://img.shields.io/badge/Homepage-Community-2563EB.svg" alt="Project homepage: NumericBridge community"></a>
</p>

<p align="center"><strong>Numerical precision comparison, diagnosis and repair</strong></p>

<p align="center">English · <a href="README.zh-CN.md">简体中文</a></p>

<p align="center"><a href="#framework-overview">Overview</a> · <a href="#getting-started">Getting started</a> · <a href="#typical-cases">Typical cases</a> · <a href="#evaluation-and-repair-history">Results and repair history</a></p>

NumericBridge is an RL-focused framework for numerical comparison, diagnosis and repair across execution stacks, with reusable skills and public community records.

## Framework overview

![NumericBridge workflow: target setup, numerical comparison, issue diagnosis, repair and optimization, regression testing, deliverables, and community knowledge retrieval](docs/images/numericbridge-overview.svg)

*The Ascend 910C / Qwen3.8-27B / verl configuration in the figure is an example. See [Typical cases](#typical-cases) for measured results and verified coverage.*

NumericBridge starts from the target hardware, model, execution stack, workload and validation conditions. It compares numerical results, diagnoses observed differences, repairs confirmed defects when requested, and re-runs the original workload in the user's environment. Depending on the task, deliverables include adapted implementations, runtime configurations, patches, and validation reports with actual coverage and evidence.

Community records help retrieve known issues, affected versions, alternative solutions, candidate fixes and validation evidence. Findings suitable for public sharing can be contributed through a pull request for review and publication; uploading is optional.

## Repository

The main deliverable is **Skills + Tools**: skills guide an agent through the task;
tools execute commands, collect and compare results, and retain evidence. Load a
skill for the next missing capability; no research runner or website is required.

| Directory | Purpose |
|---|---|
| [skills/](skills/precision-compare/SKILL.md) | Adaptation, numerical comparison, diagnosis, repair and original-workload verification |
| [tools/](tools/README.md) | Shared execution, resource verification and evidence tools |
| [community/](community/README.md) | Website, public results, retrieval and record validation |
| [scripts/](scripts/README.md) | Installation, export and repository maintenance commands |
| [docs/](docs/), [reports/](reports/README.md) | Guides and final numerical reports |

Comparison-specific arithmetic and collectors stay with their skill. Put tools
in `tools/` only when they are shared. Community retrieval selects known issues
and candidate solutions; the user's actual configuration still needs verification.

## Workflow

```text
User goal → Reuse environments and evidence → Compare and diagnose → Repair when needed → Verify the original workload
```

Start at the first missing stage and execute only the required work. Deliver
metrics, coverage, configurations or patches and verification evidence. Component
completion does not establish full RL correctness; undeclared numerical acceptance
remains null. Optional agent-improvement experiments and their snapshots are retained locally
and are not distributed in this repository.

## Getting started

See the [repository layout and tool ownership (Chinese)](docs/project-architecture.md).

The [RL diagnosis and repair proposal (Chinese)](docs/rl-precision-workflow-proposal.md)
plans reusable case workflows on top of this repository’s existing capabilities.

Export the skills side by side into a fresh client skill directory. The exporter
includes the shared tools so Compare remains standalone:

```sh
python -B -m scripts install --output /path/to/client/skills
```

See [portable installation](tools/README.md#portable-skills) when updating.
Load a component for the next missing capability:

| Skill | When to use it and what it delivers |
|---|---|
| [Framework Adapt](skills/framework-adapt/SKILL.md) | Native execution is missing: deploy or adapt the target workload and verify execution |
| [Precision Compare](skills/precision-compare/SKILL.md) | Collection or comparison is missing: return numerical metrics, coverage and evidence |
| [Precision Repair](skills/precision-repair/SKILL.md) | Diagnosis or repair is requested: establish correctness evidence, patch confirmed defects and verify the original workload |
| [Operator Optimize](skills/operator-optimize/SKILL.md) | Placeholder; not implemented |

Start from existing workload commands, inputs/state, collectors and results. Load only the component and references needed for missing work; shared tools can be called directly without loading their owning skill. `precision-compare` is the renamed comparison package previously called `precision-skill`; replace the old installed directory when updating.

Provide the requested outcome, workload or existing artifacts, target execution stacks and resource scope. Continue in the same workspace as additional capabilities become necessary; reuse execution, inputs/state, collectors and results across components. [Execution tools and the resource catalog](skills/precision-compare/references/comparison.md#shared-tasks) support reusable command chains. Repair remains task-selected; code changes rerun affected verification.

### Example requests

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

### Qwen3.8-Flash-Next

The [text precision report](reports/qwen38-precision-report.md) and [verified evidence](community/site/evidence/qwen38-precision-20261106.json) cover 26 native core captures, full 32K inference, LoRA, actual Trainer/SFTTrainer, fixed-trajectory TRL/verl components and MoE/QSA interventions. Final inference compares GPU automatic placement with NPU TP8; other families retain their actual configurations. Both GPU serving engines completed, while native NPU vLLM/SGLang initialization failures remain explicit numerical-coverage gaps. Acceptance thresholds are unspecified.

### DeepSeek-V4.1-Flash-1017

The [precision report](reports/deepseek41-precision-report.md) and [verified evidence](community/site/evidence/deepseek41-precision-20261106.json) compare the pinned original GPU TP4 implementation with an explicitly adapted BF16 NPU TP8/EP8/offload48 configuration. Coverage is 191/275 paired tensor observations and all 3,168 target scores; actual NPU 32K full and exact split requests complete. Training, selected operator scopes and rejected native chains retain numerical gaps. Weight-conversion checks do not establish equivalent backend arithmetic; acceptance thresholds are unspecified.

The separately adapted [public result metrics](community/site/data/deepseek41-adapted-core-20261109.json) adds 84 cache/packed-mask observations, actual global array reductions and 380 fresh frozen-backbone forwards for output-head-only LoRA. The [public result metrics](community/site/data/deepseek41-full40-attention-20261109.json) independently replays 480 rank-zero layer-forwards over 1,920 retained query rows using actual native operands. Both are available on the [writing page](community/site/paper.html). The original 191/275 ledger remains frozen; FP64 attention replay does not emulate native rounding. The newer [public result metrics](community/site/data/deepseek41-paired-output-head95-20261109.json) adds 95 same-coordinate calls per side (83 updates and 12 reserved evaluations), independently replayed formulas and exact within-runtime continuation/repeats. Initial A is near-equal rather than byte-exact across runtimes; later optimizer states follow independent trajectories. SGLang and vLLM service numerical reviews are complete, with the vLLM Core77 lifecycle limitation retained. These supplements do not establish full-model backward or numerical acceptance.

- **DeepSeek-V4.1-Flash-1017 — reviewed service evidence (9 November 2026 UTC).**
  The [English report](reports/deepseek41-services-final-20261109/report.md) covers all 277 SGLang tasks across five Engine lifetimes and 216 vLLM TP8/EP8 jobs in one Engine on eight H200 GPUs. Original-host raw mathematics and independent portable reviews are complete; all 820 formal vLLM full-vocabulary rows replay locally. Original GPU **191/275**, original NPU **269/275** and the separate adapted paired **275/275** inventories remain distinct; 3,168 target coordinates are retained. The 95-call output-head study keeps the backbone frozen.
  Core77's final restoration receipt is missing: the original strict lifecycle review remains failed, although all eight data-worker finals and exact namespace cleanup are verified. Twelve scientific gaps remain explicit. Full-vocabulary normalization, matched causal histories and null acceptance thresholds are retained; GPU is not a mathematical oracle. The [service verification report](reports/deepseek41-services-final-20261109/report.md) provides the current report, scientific figure and precise JSON/CSV pointers. Historical ZIPs, raw tensors and complete execution logs remain local. No publication or full-backbone training claim is made.

[HTTP128 metrics](community/site/data/deepseek41-http128-20261109.json).

### Precision repair

Qwen3.5-9B-Base: repaired zero-centered RMSNorm parameter semantics in a pinned historical stack. Verification covered native NPU operator forward/backward, checkpoint round trips, and the original TP4 model forward with unchanged loaded parameter bytes.

Additional historical backports were verified using small native CPU workloads:

| Library / pinned version | Repair | Verification |
|---|---|---|
| TRL 0.7.10 | IPO completion logprob averaging | Native DPOTrainer loss, all gradients and one SGD update; 6 regressions |
| PEFT 0.17.1 | Duplicate LoRA scaling in SVD adapter combination | Forward, input gradients, checkpoint reload and merge; 24 workloads |
| Transformers 4.51.3 | Qwen2 attention mask ignores disabled sliding window | Eager/SDPA full forward and dynamic/static cache; 18 workloads and 16 mask cases |

The [repair history](https://satsuki2486441738.github.io/NumericBridge/) records affected and upstream-fixed revisions, patches, before/after metrics and verification scope.

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

The [comparison board](https://satsuki2486441738.github.io/NumericBridge/) reads published evaluation and repair records. The [history archive](https://satsuki2486441738.github.io/NumericBridge/archive.html) retains inference/SFT/RL settings, numerical metrics, coverage and evidence, alongside library-grouped fixes and verification. Records and public evidence are maintained in [community/site/](community/site/README.md).

The [RSI dashboard](https://satsuki2486441738.github.io/NumericBridge/rsi.html) shows current runs, candidate changes, independent evaluation outcomes and phase-specific time/token usage, including baseline reuse.

The [results page](community/site/paper.html) provides public reports, scientific figures, [JSON](community/site/data/paper-results.json) and a [CSV numeric index](community/site/data/paper-results.csv). Published results retain source hashes, actual configurations, coverage, failures and unverified scopes. Numeric fields are not independent samples. Historical ZIPs, raw tensors and complete execution logs remain local with no online download. The [local evidence index](community/site/data/local-evidence.json) records archive names, sizes and SHA256 hashes. Byte integrity does not imply numerical acceptance.

Public repository scope: [publication policy](docs/publication.md). Keep final results and reproducibility metadata public; retain intermediate execution artifacts locally.

Developer regression files in `tests/` are retained locally and Git-ignored. Public CI runs repository/history validation and the clean-install smoke check, without depending on that local suite.
