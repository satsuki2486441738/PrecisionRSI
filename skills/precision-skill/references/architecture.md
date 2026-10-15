# Package and workspace

`SKILL.md` routes to methods in `references/` and [offline tools](tools.md) in `scripts/`. Load only relevant references. The agent coordinates and interprets; native collectors save observations; offline validators check comparability and compute metrics.

Keep case configuration, implementation and history outside the installed skill:

```text
projects/<project>/
  environments/<side>/<version>/  # Declarations and locks
  cases/<case>/
    contract.json                # Shared conditions and metrics
    <side manifests>             # Execution evidence references
    README.md / WORK_LOG.md       # Current guide and brief history
    scripts/ / tests/            # Case tooling
    runs/<side>-<task>-<UTC>/     # Immutable observations and logs
```

Use side `gpu`, `npu` or `compare`; create runs exclusively and preserve historical bytes/paths. Reference models, data and environments at their resource locations. Move case code into shared tooling only when reuse warrants it; report the actual collector/comparator's coverage.
