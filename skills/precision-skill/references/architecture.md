# Package and workspace

The installable skill separates comparison methods from execution integration and case evidence:

| Responsibility | Location |
|---|---|
| Task state, observations and metrics | [Inference](inference.md), [SFT/RL](training.md) and [comparison](comparison.md). |
| Native collection integration | [Inference engines](engines.md), [RL frameworks](rl-frameworks.md) and [training backends](training-backends.md). |
| Artifact interfaces and arithmetic | [Offline tools](tools.md) in `scripts/`. |
| Configuration and execution history | Private project environments and cases. |

`SKILL.md` routes to the current task. The agent coordinates execution and interpretation, native collectors save observations, and offline tools validate artifacts and compute metrics. Load integration details when implementing or reviewing that layer.

## Case ownership

```text
projects/<project>/
  environments/<side>/<version>/  # Reusable declarations and locks
  cases/<case>/
    contract.json                # Shared conditions and metrics
    <side manifests>             # Environment and execution references
    README.md / WORK_LOG.md       # Current guide and execution history
    scripts/ / tests/            # Case collectors, adapters and regressions
    runs/<side>-<task>-<UTC>/     # Immutable observations and logs
```

Use side `gpu`, `npu` or `compare` and create run directories exclusively. Keep model, data and environment resources at their resource locations; retain historical artifact bytes and paths.

## Reuse

Keep model- and case-specific collectors and conversion maps with their case. Promote code when multiple cases share the same responsibility and interface. Incorporate reusable findings into the relevant method or integration flow; retain incident chronology and local workarounds in case logs. Each case reports the coverage established by its actual execution and comparison.
