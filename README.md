# precision-skill

English | [简体中文](README.zh-CN.md)

An [Agent Skill](https://agentskills.io/home) for comparing numerical results across NVIDIA GPU and Ascend NPU execution stacks.

## Use

Copy [skills/precision-skill](skills/precision-skill/) into your client's skill location, retaining the directory name. Provide the task, two stacks and available artifacts. The agent reuses existing configurations and approvals, aligns conditions, collects observations, validates coverage and reports numerical differences.

| Task | Method |
|---|---|
| [Inference](skills/precision-skill/references/inference.md) | Compare next-token top-100 unions and aggregate tails: KL, top-1 agreement, probability errors and coverage. |
| [SFT](skills/precision-skill/references/training.md#sft) | Same-state single step: all valid target-token scores, full gradients and full actual parameter updates. |
| [RL](skills/precision-skill/references/training.md#rl) | Fixed trajectories and algorithm-specific dependencies, objectives, gradients and updates. |

GPU is the reference. Numerical diagnostics without predefined thresholds are not acceptance passes. Read only the current task's reference; case settings, research rationale and execution history stay outside the method descriptions.

For vLLM and SGLang, the [engine collection guide](skills/precision-skill/references/engines.md) covers native prefill and fixed-history cached decode, raw-score semantics and complete distribution collection. Engine adapters remain case-specific; this guide does not imply device validation.

For RL with TRL or verl, the [framework guide](skills/precision-skill/references/rl-frameworks.md) defines fixed-trajectory replay and native optimizer-step collection. Default metrics cover action scores, losses, gradients and actual updates; algorithm-specific diagnostics are conditional. Framework collectors remain case-specific.

## Tooling and layout

The [offline tools](skills/precision-skill/references/tools.md) validate artifacts and compare top-100 distributions or full gradients/actual updates using NumPy and bounded-memory reads. The existing standard-library target-score comparator remains available. Collectors, format adapters and task-specific loss reconstruction stay with each case; no tool claims cross-device agreement from a self-check.

- [SKILL.md](skills/precision-skill/SKILL.md): pipeline and reference routing.
- [Comparison](skills/precision-skill/references/comparison.md): alignment, formulas and result interpretation.
- [Environment](skills/precision-skill/references/environment.md): shared releases and execution-associated evidence.
- [Workflow](skills/precision-skill/references/workflow.md): device runs, retries and handoffs.
- [Architecture](skills/precision-skill/references/architecture.md): collector responsibilities and case layout.

Private `projects/` own case configurations, collectors and runs. Weights, data and installed environments remain at their resource locations. Technical skill documentation is maintained in English; the root guides are bilingual.

Run maintainer tests in an existing Python environment with NumPy, from the repository root: `python -B -m unittest discover -s tests -v`.
