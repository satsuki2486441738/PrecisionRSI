---
name: precision-skill
description: Diagnose numerical differences between GPU and NPU execution stacks for inference, SFT, or RL. Use when comparing model scores, losses, gradients, or parameter updates, validating an execution handoff, or investigating a precision discrepancy. Align shared dependencies, establish comparable inputs and states, and report measured differences with their coverage.
---

# Precision diagnosis

Compare native execution stacks under a shared numerical contract. Either side can run first; artifacts support offline comparison without switching repeatedly between devices.

## Start here

1. Establish the user's question, task, two stacks, available artifacts, and execution scope. Discover facts from existing records; ask together for decisions that affect the next stage. Follow [workflow](references/workflow.md) for budgets, task handoffs, and stopping conditions.
2. Inventory the relevant artifacts and reuse matching inputs, resources, and results. Follow [environment and resource evidence](references/environment.md) to align shared library releases and associate each run with its environment. Obtain explicit approval before collecting or comparing with different shared versions.
3. Freeze the comparison conditions and required observations. Read the relevant task reference:
   - [Inference](references/inference.md): fixed-history scoring, serialization, and cache comparisons.
   - [Training](references/training.md): SFT objectives, gradients, actual updates, or RL trajectory dependencies.
4. Prepare the smallest native execution that can produce the required evidence. Validate input replay, observations, references, and every expected rank before increasing scale. Preserve platform kernels and record precision by stage.
5. Check comparability before calculating differences. Use [comparison and handoff](references/comparison.md) for the available schema, evidence checklist, metrics, and reporting rules. Add internal observations when an anomaly or a specific user question warrants them.
6. Report execution, comparability, numerical results, coverage, and unresolved blockers separately. Link the run evidence and record the next useful action in the case work log.

## Implemented tool

`scripts/compare_scores.py` is a Python standard-library comparator for schema-v2 full-forward token scores. It accepts saved artifacts and performs no model execution. It reports token-weighted NLL, delta NLL, MAE, and P99 with `thresholds: null`.

SFT, RL, cache, and tensor relative-L2 comparisons use task-specific collectors and validators prepared for the case. The references describe their required semantics; the bundled comparator implements full-forward scoring.

## Organization

Read [architecture](references/architecture.md) when creating collectors, environments, or cases. Keep model weights, source datasets, and environment installations at their resource locations. Shared resources are checked during preparation and bound to actual loads on reuse.
