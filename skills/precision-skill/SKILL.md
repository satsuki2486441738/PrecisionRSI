---
name: precision-skill
description: Compare GPU and NPU numerical results for inference, SFT, or RL. Align inputs and execution conditions, collect comparable observations, compute precision metrics, and localize discrepancies.
---

# Precision diagnosis

Compare native execution stacks under common inputs and numerical conditions. GPU is the reference; either side may execute first and exchange artifacts offline.

## Pipeline

1. Reuse the user's scope, existing configurations, artifacts and approvals. Resolve facts from records; ask only for a missing decision that materially changes the next action. Do not turn this pipeline into a questionnaire.
2. Align model/data identities, shared implementation releases, precision policy and task state. Read [environment](references/environment.md) when preparing or reviewing an environment.
3. Collect the observations for the current task: [inference](references/inference.md) or [SFT/RL](references/training.md). Read only the applicable task section; retain native kernels and fusion.
4. Validate complete paired coverage and compute the task's metrics using [comparison](references/comparison.md). Investigate internals only when an observed discrepancy or the user's question requires it.
5. Report the outcome, key metrics, coverage and evidence location. Without predefined thresholds, report diagnostics rather than a pass.

Use [workflow](references/workflow.md) for device execution/retries and [architecture](references/architecture.md) when adding a collector or case. Case-specific models, versions, budgets, paths, run history and research rationale belong outside this skill's method descriptions.

## Available tool

[scripts/compare_scores.py](scripts/compare_scores.py) implements schema-v2 full-forward **target-token** scoring: NLL, delta NLL, MAE and P99, with null thresholds. Top-100 distributions, SFT, RL and cache comparisons need case-specific collectors/validators. Read the module's schema docstring only when using or adapting that interface.
