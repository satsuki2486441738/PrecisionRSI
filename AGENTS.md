# Repository maintenance

## Layout

- `skills/precision-skill/` is the installable package. Keep routing and common collection steps in `SKILL.md`, the three task recipes and `comparison.md` in `references/`, and artifact interfaces in script docstrings. Optional internal sampling and its msProbe capability summary live in `references/layerwise.md`. Task recipes own capture requirements, framework integration and metric selection. Define shared training stages in SFT, distribution metrics in inference, and common coordinate/metric/tool rules in `comparison.md`.
- Maintain matching English `README.md` and Chinese `README.zh-CN.md`. Technical documentation and maintenance instructions use English. Follow the Agent Skills specification.
- `skills/precision-skill/examples/` contains concise runnable artifact examples. Extract task-independent operations into composable tools with explicit interfaces and meaningful regressions; keep task-specific settings in adapters and environments.
- `tests/` contains public tool regressions. Artifact tools require NumPy; the target-score comparator uses the standard library.
- Private `projects/<project>/environments/` owns reusable environment declarations. Cases own configuration, collectors, tests and execution artifacts. Keep case-specific code there; extract shared code when reused.
- Keep current guides focused on task semantics and tool usage. Work logs and `runs/` own dated decisions, one-off execution failures and results; resolve tool defects in code and regressions.

## Changes

Confirm the approach before editing and change at most three files per batch. For a bug fix, reproduce the behavior temporarily, add a meaningful regression and remove the temporary reproducer. Preserve permanent tests.

Create new runs exclusively as `<side>-<task>-<UTC YYYYmmddTHHMMSSZ>`, using the task's side ID or `compare`. Keep model/data/environment resources in place and preserve historical run bytes and paths. Use a new run for later validation. For undated migrations, use `<side>-<task>-migrated-<UTC migration timestamp>` and retain the path mapping.

## Validation

Run `python -B -m unittest discover -s tests -v` from the repository root. Validate skill metadata, local links/anchors and changed script behavior. For documentation edits, check documented fields and commands against their owning scripts.
