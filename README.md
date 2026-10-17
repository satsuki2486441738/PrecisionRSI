# precision-skill

English | [简体中文](README.zh-CN.md)

An [Agent Skill](https://agentskills.io/home) for comparing numerical results across NVIDIA GPU and Ascend NPU execution stacks.

## Use

Copy [skills/precision-skill](skills/precision-skill/) into your client's skill location, retaining the directory name. Provide the task, two stacks and available artifacts. The agent reuses existing configurations and approvals, aligns conditions, collects observations, validates coverage and reports numerical differences.

| Task | Method |
|---|---|
| [Inference](skills/precision-skill/references/inference.md) | Compare next-token top-100 unions and aggregate tails: KL, top-1 agreement, probability errors and coverage. |
| [SFT](skills/precision-skill/references/training.md#sft) | Same-state single step: all valid target-token scores, full gradients and full actual parameter updates; LoRA adapter scope and frozen-base evidence; cold/restored optimizer-state baselines and regression reuse. |
| [RL](skills/precision-skill/references/training.md#rl) | Fixed trajectories and algorithm-specific dependencies, objectives, gradients and updates. |

GPU is the reference. The default task delivers collection, comparison, coverage and evidence. Numerical differences are findings, and the user judges acceptance; they do not prevent subsequent valid collection. Precision remediation is a separate task when requested. Read only the current task's reference; case settings, research rationale and execution history stay outside the method descriptions.

Choose the integration reference for the layer being collected:

| Layer | Reference |
|---|---|
| Inference engine | [vLLM / SGLang](skills/precision-skill/references/engines.md): prefill, fixed-history cached decode and score semantics. |
| RL framework | [TRL / verl](skills/precision-skill/references/rl-frameworks.md): trajectory preparation, native objectives and update boundaries. |
| Training backend | [FSDP / ZeRO / Megatron-Core / MindSpeed](skills/precision-skill/references/training-backends.md): native storage, optimizer stages and logical tensor mapping. |

Collectors and execution evidence belong to cases. These references describe methods; each case records its actual coverage and validation status.

## Tooling and layout

The [offline tools](skills/precision-skill/references/tools.md) validate artifacts and compare top-100 distributions or full gradients/actual updates using NumPy and bounded-memory reads. The existing standard-library target-score comparator remains available. Collectors, format adapters and task-specific loss reconstruction stay with each case; no tool claims cross-device agreement from a self-check.

- [SKILL.md](skills/precision-skill/SKILL.md): pipeline and reference routing.
- [Comparison](skills/precision-skill/references/comparison.md): alignment, formulas and result interpretation.
- [Environment](skills/precision-skill/references/environment.md): shared releases and execution-associated evidence.
- [Workflow](skills/precision-skill/references/workflow.md): device runs, retries and handoffs.
- [Architecture](skills/precision-skill/references/architecture.md): collector responsibilities and case layout.

Private `projects/` own case configurations, collectors and runs. Weights, data and installed environments remain at their resource locations. Technical skill documentation is maintained in English; the root guides are bilingual.

Run maintainer tests in an existing Python environment with NumPy, from the repository root: `python -B -m unittest discover -s tests -v`.
