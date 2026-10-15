# Package and workspace

The installable package is `skills/precision-skill/`: `SKILL.md` routes tasks, `references/` describes methods, and `scripts/compare_scores.py` implements the target-score comparator. Load references only when needed. Repository README files and maintainer tests stay outside the installed skill.

| Component | Responsibility |
|---|---|
| Agent | Reuse context, align conditions, organize execution and interpret results. |
| Native collector | Replay inputs/state and save actual observations and execution evidence. |
| Offline validator/comparator | Check supported conditions and complete coverage, then compute metrics. |

Keep case-specific resources in the execution workspace:

```text
projects/<project>/
  environments/<side>/<version>/  # Environment declarations and locks
  cases/<case>/
    contract.json                # Shared conditions and metrics
    <side manifests>             # Paths to execution evidence
    README.md / WORK_LOG.md       # Current guide and brief history
    scripts/ / tests/            # Case collectors and validators
    runs/<side>-<task>-<UTC>/     # Immutable observations and logs
```

Use side `gpu`, `npu` or `compare`; create run directories exclusively. Keep model weights, source data and installed environments at their existing resource locations. Reference them through preparation and actual-load evidence rather than copying them into every run.

Move case code into shared tooling only when reuse warrants it. A documented method is not an implemented capability: report what the actual collector and comparator cover. Preserve historical results under their original scope.
