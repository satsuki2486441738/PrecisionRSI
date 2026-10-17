---
name: precision-skill
description: Compare GPU and NPU numerical results for inference, SFT, or RL. Align inputs and execution conditions, collect comparable observations, compute precision metrics, and report differences with coverage and evidence.
---

# Precision comparison

Compare native execution stacks under common inputs and numerical conditions. GPU is the reference; either side may execute first and exchange artifacts offline.

## Pipeline

1. Reuse the user's scope, existing configurations, artifacts and approvals. Resolve facts from records; ask only for a missing decision that materially changes the next action.
2. Align model/data identities, shared implementation releases, precision policy and task state. Read [environment](references/environment.md) when preparing or reviewing an environment.
3. Collect the observations for the current task: [inference](references/inference.md) or [SFT/RL](references/training.md). Read common rules and the applicable task section; retain native kernels and fusion.
4. Validate complete paired coverage and compute the task's metrics using [comparison](references/comparison.md). Investigate internals when localization is part of the agreed task.
5. Report the outcome, key metrics, coverage and evidence location. Numerical differences are findings; continue valid collection stages without treating numerical acceptance as a prerequisite. The user judges acceptance.

Precision remediation is a separate task when requested. Repair collection and comparison failures within the agreed execution scope.

Use [workflow](references/workflow.md) for device execution/retries and [architecture](references/architecture.md) when adding a collector or case. Case-specific models, versions, budgets, paths, run history and research rationale belong outside this skill's method descriptions.

## Available tool

Use [offline tools](references/tools.md) for artifact validation, top-100 distributions, full training tensors or legacy target-token scores. Collectors and format adapters remain case-specific. Read a script's schema docstring only when implementing its input interface.
