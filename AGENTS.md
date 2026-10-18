# Repository maintenance

## Layout

- `skills/precision-skill/` is the installable package. Keep task instructions in `SKILL.md`, focused methods in `references/`, and artifact interfaces in script docstrings.
- Maintain matching English `README.md` and Chinese `README.zh-CN.md`. Technical documentation and maintenance instructions use English. Follow the Agent Skills specification.
- `tests/` contains public tool regressions. Artifact tools require NumPy; the target-score comparator uses the standard library.
- Private `projects/<project>/environments/` owns reusable environment declarations. Cases own configuration, collectors, tests and execution artifacts. Keep case-specific code there; extract shared code when reused.
- Keep current guides procedural. Work logs and `runs/` own dated decisions, failures and results.

## Changes

Confirm the approach before editing and change at most three files per batch. For a bug fix, reproduce the behavior temporarily, add a meaningful regression and remove the temporary reproducer. Preserve permanent tests.

Create new runs exclusively as `<side>-<task>-<UTC YYYYmmddTHHMMSSZ>`, using the task's side ID or `compare`. Keep model/data/environment resources in place and preserve historical run bytes and paths. Use a new run for later validation. For undated migrations, use `<side>-<task>-migrated-<UTC migration timestamp>` and retain the path mapping.

## Validation

Run `python -B -m unittest discover -s tests -v` from the repository root. Validate skill metadata, local links/anchors and changed script behavior. For documentation edits, check documented fields and commands against their owning scripts.
