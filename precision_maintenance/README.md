# Repository maintenance

This package owns collection, integration planning, public projections, statistics
export, publication guardrails, skill export and local validation. CI calls these
modules; reusable business logic no longer lives in workflow YAML or `.github/`
scripts. Modules load on demand. RSI snapshots read retained research receipts and
do not import or run the RSI engine.

```sh
python -B -m precision_maintenance --help
python -B -m precision_maintenance validate
python -B -m precision_maintenance validate-immutability --base origin/main
python -B -m precision_maintenance plan-history --help
python -B -m precision_maintenance export-statistics --help
python -B -m precision_maintenance install --output /tmp/exported-skills
python -B -m precision_maintenance smoke-install
```

`validate` checks skill metadata and tracked Markdown links. The community package
owns catalog validation: `python -B -m precision_community validate`.
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
| `validate.py` | `python -B -m precision_maintenance validate` |
| `validate_history.py` | `python -B -m precision_maintenance validate-history` |
| `validate_differences.py` | `python -B -m precision_maintenance validate-differences` |
| `validate_immutability.py` | `python -B -m precision_maintenance validate-immutability` |
| `collect_differences.py` | `python -B -m precision_maintenance collect-differences` |
| `export_paper_results.py` | `python -B -m precision_maintenance export-paper` |
| `export_statistics.py` | `python -B -m precision_maintenance export-statistics` |
| `plan_history_integration.py` | `python -B -m precision_maintenance plan-history` |
| `prepare_public_results.py` | `python -B -m precision_maintenance prepare-public` |
| `rsi_snapshot.py` | `python -B -m precision_maintenance rsi-snapshot` |
| `smoke_install.py` | `python -B -m precision_maintenance smoke-install` |

Install maintenance dependencies with `python -m pip install -r requirements-dev.txt`.
Final reports live in [reports/](../reports/README.md); GitHub platform configuration
alone belongs in `.github/`. Historical source-path strings inside immutable
published evidence remain unchanged and describe the original capture.
