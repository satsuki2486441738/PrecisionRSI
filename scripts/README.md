# Maintainer scripts

These scripts handle collection, integration planning, public projections, statistics
export, publication guardrails, skill export and local validation. CI calls these
modules; reusable business logic no longer lives in workflow YAML or `.github/`
scripts. Modules load on demand. RSI snapshots read retained research receipts and
do not import or run the RSI engine.

```sh
python -B -m scripts --help
python -B -m scripts validate
python -B -m scripts validate-immutability --base origin/main
python -B -m scripts plan-history --help
python -B -m scripts export-statistics --help
python -B -m scripts install --output /tmp/exported-skills
python -B -m scripts smoke-install
```

`validate` checks skill metadata and tracked Markdown links. The community package
owns catalog validation: `python -B -m community.records validate`.
`smoke-install` requires NumPy and Torch, performs a fresh skill export and checks
the CPU gradient/repair workflow. No model download or accelerator is used.

Additional commands: `collect-differences`, `prepare-public`, `export-paper`,
`rsi-snapshot`, `validate-history`, `validate-differences`. The selected module owns
its arguments and behavior. Retired `.github/*.py` entrypoints are replaced by
the module commands below. Read the corresponding module docstring
for the exact contract; [history integration](../docs/history-integration.md)
retains the planning/publication distinction.

Repository paths resolve against the source checkout, independently of caller cwd.
Collectors and snapshot refresh can access external sources **only when explicitly
invoked**; import and validation do not refresh data. Exporters retain their
existing scope, privacy limits and no-overwrite behavior. No public projection
upgrades partial scientific evidence into numerical acceptance.

## Migrating old script commands

Run from the repository root. Replace the old script prefix and retain its flags:

| Retired `.github/` script | Supported command |
|---|---|
| `validate.py` | `python -B -m scripts validate` |
| `validate_history.py` | `python -B -m scripts validate-history` |
| `validate_differences.py` | `python -B -m scripts validate-differences` |
| `validate_immutability.py` | `python -B -m scripts validate-immutability` |
| `collect_differences.py` | `python -B -m scripts collect-differences` |
| `export_paper_results.py` | `python -B -m scripts export-paper` |
| `export_statistics.py` | `python -B -m scripts export-statistics` |
| `plan_history_integration.py` | `python -B -m scripts plan-history` |
| `prepare_public_results.py` | `python -B -m scripts prepare-public` |
| `rsi_snapshot.py` | `python -B -m scripts rsi-snapshot` |
| `smoke_install.py` | `python -B -m scripts smoke-install` |

Install maintenance dependencies with `python -m pip install -r requirements-dev.txt`.
Final reports live in [reports/](../reports/README.md); GitHub platform configuration
alone belongs in `.github/`. Historical source-path strings inside immutable
published evidence remain unchanged and describe the original capture.

## Repository layout migration

| Former checkout entrypoint | Current entrypoint |
|---|---|
| `python -m precision_maintenance ...` | `python -m scripts ...` |
| `python -m precision_core.record_run ...` | `python -m tools.record_run ...` |
| `from precision_core.workflow import run_workflow` | `from tools.workflow import run_workflow` |
| `python -m precision_rsi ...` | Local research only; excluded from the public checkout |

Flags and task formats retain their meanings. Update active caller commands and
export a fresh skill installation; historical plans, command receipts and version
identities remain unchanged. The old root packages are not retained as duplicate
implementations. Standalone skills continue to expose their documented script
entrypoints and bundle shared tools through the exporter.
