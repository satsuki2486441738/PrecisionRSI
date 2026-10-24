# Repository maintenance

## Layout

- Precision-Skills currently contains three installable packages: `framework-adapt`, `precision-compare` and `precision-repair`. Compare owns shared artifact and task-execution tools. The three components reuse workload state, verified resources, inputs/state and metrics across adaptation, comparison and repair. `skills/operator-optimize/` is a reserved placeholder with no implementation.

- `skills/framework-adapt/` owns native framework deployment and model/backend adaptation. Keep the entrypoint concise; retain environment declarations and framework-specific implementations with their projects and cases. Reuse comparison tools when numerical validation is needed.
- `skills/precision-repair/` is the installable repair skill. Its entrypoint owns correctness evidence, causal diagnosis, patching and original-workload verification. Reuse comparison tools from `skills/precision-compare/`; keep framework-specific fixes and experiments with their cases.
- `skills/precision-compare/` is the installable comparison skill. Keep routing and common collection steps in `SKILL.md`, the three task recipes and `comparison.md` in `references/`, and artifact interfaces in script docstrings. Optional internal sampling and its msProbe capability summary live in `references/layerwise.md`. Task recipes own capture requirements, framework integration and metric selection. Define shared training stages in SFT, distribution metrics in inference, and common coordinate/metric/tool rules in `comparison.md`.
- Maintain matching English `README.md` and Chinese `README.zh-CN.md`. Technical documentation and maintenance instructions use English. Follow the Agent Skills specification.
- `skills/precision-compare/examples/` contains concise runnable artifact examples. Extract task-independent operations into composable tools with explicit interfaces and meaningful regressions; keep task-specific settings in adapters and environments.
- `tests/` contains public tool regressions. Artifact tools require NumPy; the target-score comparator uses the standard library.
- Private `projects/<project>/environments/` owns reusable environment declarations. Cases own configuration, collectors, tests and execution artifacts. Keep case-specific code there; extract shared code when reused.
- Keep current guides focused on task semantics and tool usage. Work logs and `runs/` own dated decisions, one-off execution failures and results; resolve tool defects in code and regressions.

`site/` hosts the GitHub Pages evaluation and repair histories. Append evaluations to `site/data/evaluations.json` with peer configurations, comparison scope, measured metrics, coverage and immutable evidence. Keep repair experiments in `site/data/repairs.json` with their existing library/fix/verification hierarchy. Preserve recorded IDs, measurements and evidence; keep full execution artifacts in private cases. Run `python -B .github/validate_history.py` for history changes.

## Changes

Confirm the approach before editing and change at most three files per batch. For a bug fix, reproduce the behavior temporarily, add a meaningful regression and remove the temporary reproducer. Preserve permanent tests.

Create new runs exclusively as `<side>-<task>-<UTC YYYYmmddTHHMMSSZ>`, using the task's side ID or `compare`. Keep model/data/environment resources in place and preserve historical run bytes and paths. Use a new run for later validation. For undated migrations, use `<side>-<task>-migrated-<UTC migration timestamp>` and retain the path mapping.

For skill/no-skill benchmark iterations, reuse a verified no-skill baseline while the task and model/runtime conditions remain comparable; rerun only the skill arm. Keep baseline artifacts outside evaluator inputs, record the reused baseline identity, and retain raw usage for consistent repricing.

## Validation

Install maintenance dependencies with `python -m pip install -r .github/requirements.txt`. Run `python -B -m unittest discover -s tests -v` and `python -B .github/validate.py` from the repository root. With Torch installed, run `python -B .github/smoke_install.py` to verify clean skill installations, companion links and the CPU gradient example. Validate skill metadata, local links/anchors and changed script behavior. For documentation edits, check documented fields and commands against their owning scripts.
