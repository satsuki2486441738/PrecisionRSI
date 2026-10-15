# precision-skill

English | [简体中文](README.zh-CN.md)

An [Agent Skill](https://agentskills.io/home) for numerical diagnosis across Ascend NPU and NVIDIA GPU execution stacks. It organizes comparable inputs, environment evidence, collection, offline comparison, and focused investigation for inference, SFT, and RL.

## Package and use

The installable skill is [skills/precision-skill](skills/precision-skill/). Copy that entire directory into your agent client's skill location, retaining the directory name `precision-skill`. The package follows the [Agent Skills specification](https://agentskills.io/specification).

Start with a task such as:

> Compare one SFT step on GPU and NPU using the same model and initial state. Reuse the GPU handoff, align shared dependencies, and compare loss, sampled gradients, and actual parameter updates.

Provide the two stacks, hardware/environment access, available artifacts, and the question to answer. The agent discovers existing facts and asks for decisions needed by the current stage. Either side can execute first; the other can consume its handoff offline.

## Implemented scope

The bundled [Python comparator](skills/precision-skill/scripts/compare_scores.py) uses only the standard library. Schema v2 / core 2.0.0 validates saved full-forward token scores and reports token-weighted NLL, delta NLL, MAE, and global linear P99. Its current threshold mode is `thresholds: null`.

The workflow also covers SFT loss/gradient/update semantics and RL fixed-trajectory dependencies. Those tasks use case-specific collectors and validators; the bundled scoring core does not calculate training or cache comparisons. Reports distinguish execution, comparability, numerical results, and observed coverage.

## Evaluation protocols

The [inference protocol](skills/precision-skill/references/inference.md#next-token-distribution-protocol-top100-union-v1) now specifies `top100-union-v1`: each side selects its top 100 next-token candidates at the same fixed-history positions, then provides probabilities for the union and its remaining tail mass. Primary diagnostics are GPU-to-NPU coarsened KL and top-1 agreement, supplemented by probability RMSE/max error, top-100 overlap, probability coverage, and auxiliary target-token NLL. Thresholds remain unset. This protocol is documented; distribution collection and comparison are not yet implemented by the bundled tool, and historical target-score reports do not satisfy it.

The [anchored SFT protocol](skills/precision-skill/references/training.md#sft), `sft`, requires every valid target token, full gradients and full actual parameter updates for one same-state step. It reports target-logprob MAE/P99/max and loss differences, plus global and per-tensor relative L2, cosine and maximum absolute error for gradients/updates. Existing sampled runs are smoke tests; full-coverage collection/comparison still needs implementation. RL recommendations additionally cover fixed actions, old/reference logprobs, rewards/advantages, ratio/clipping records and algorithm-specific critic or group data; RL is not yet anchored. The reference cites TRL and msprobe and distinguishes numerical agreement from learning effectiveness. Bind execution details before collection; thresholds remain unset.

Shared implementation libraries use the same upstream release, including the patch version. Platform build strings and extensions are recorded separately. A different-version comparison requires explicit approval of the versions and scope. Existing runtime inventories or associated locked-installation and launch records can establish the environment; missing a dedicated inventory file does not automatically require a rerun.

## Layout

```text
README.md / README.zh-CN.md        English and Chinese project guides
AGENTS.md / CLAUDE.md              Repository maintenance instructions
skills/precision-skill/
  SKILL.md                        Discovery metadata and task routing
  references/                     Focused English technical guidance
  scripts/compare_scores.py       Offline scoring comparator
tests/                            Maintainer regression tests
```

Private projects contain their own environments, contracts, collectors, and runs. Model weights, source data, and installed environments remain at shared resource locations. Preparation records are reused with actual load bindings.

## References

| Document | Use |
|---|---|
| [Skill entry](skills/precision-skill/SKILL.md) | Task selection and workflow entry. |
| [Workflow](skills/precision-skill/references/workflow.md) | Scope, budgets, collection, and task handoffs. |
| [Environment](skills/precision-skill/references/environment.md) | Shared versions, lockfiles, run association, and resource reuse. |
| [Inference](skills/precision-skill/references/inference.md) | Fixed-history scoring and cache semantics. |
| [Training](skills/precision-skill/references/training.md) | SFT steps, RL dependencies, and state divergence. |
| [Comparison](skills/precision-skill/references/comparison.md) | Schema, evidence, formulas, and report states. |
| [Architecture](skills/precision-skill/references/architecture.md) | Package structure, responsibilities, and workspace ownership. |

Run maintainer tests from the repository root:

```bash
python -B -m unittest discover -s tests -v
```

The two README files describe the same scope. Skill instructions, technical references, and maintenance documentation are maintained in English.
