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

## Tooling and layout

The bundled [comparator](skills/precision-skill/scripts/compare_scores.py) implements schema-v2 full-forward **target-token** NLL, delta NLL, MAE and P99 with null thresholds. Top-100, training and cache comparisons use case-specific tooling; the documentation does not imply those features exist in the bundled comparator.

- [SKILL.md](skills/precision-skill/SKILL.md): pipeline and reference routing.
- [Comparison](skills/precision-skill/references/comparison.md): alignment, formulas and result interpretation.
- [Environment](skills/precision-skill/references/environment.md): shared releases and execution-associated evidence.
- [Workflow](skills/precision-skill/references/workflow.md): device runs, retries and handoffs.
- [Architecture](skills/precision-skill/references/architecture.md): collector responsibilities and case layout.

Private `projects/` own case configurations, collectors and runs. Weights, data and installed environments remain at their resource locations. Technical skill documentation is maintained in English; the root guides are bilingual.

Run maintainer tests from the repository root: `python -B -m unittest discover -s tests -v`.
