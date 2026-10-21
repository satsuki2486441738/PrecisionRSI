# Precision history website

GitHub Pages serves this directory. Preview from the repository root with `python3 -m http.server 8000 --directory site`, then open `http://localhost:8000`.

## Record an evaluation

The **Evaluations** tab reads `data/evaluations.json` and groups experiments by inference, SFT and RL. Each experiment has a stable ID/date, model revision, common conditions, comparison variables, actual per-side hardware/libraries/settings, and an immutable archive revision. A comparison references two declared side IDs and owns its scope, numerical metrics and observed/expected coverage. An experiment can contain multiple steps or execution modes without becoming multiple history entries.

Metric `value` is the symmetric comparison result. Optional `side_values` records each compared side's value in the same unit; optional `coordinate` identifies the maximum error. Keep finite numbers at recorded precision. Equal-array counts are measured results; coverage counts collected/paired observations. Set `acceptance: null` until a numerical acceptance decision is recorded. `status: "completed"` means collection/comparison finished.

Append concise public evidence under `evidence/` and keep native arrays/checkpoints in the existing artifact store. Preserve recorded IDs, values and immutable provenance. The initial entries cover same-H200 BF16/FP16 inference, native cross-platform TRL LoRA SFT and same-platform checkpoint restoration with peer comparison. New frameworks and hardware require data entries, not UI changes.

Links use `#evaluations` and `#evaluation-<id>`. Existing repair/fix links still select the Repairs tab. Both tabs support keyword, library, model and hardware filters; evaluations also support task filtering and comparison selection. JSON downloads contain the selected archive.

Evaluation charts use the Repair tab's gray/green palette. Paired bars show the
actual declared sides and a shared linear domain including zero; they are not
before/after repairs. Signed values retain their signs. Error-magnitude bars share a scale
only within the selected comparison, metric family and unit. JS divergence is
scaled separately from target-score errors. Cosine and agreement charts use
their mathematical domains; unrecognized metrics retain their numerical values
without an invented range. Coverage bars show collected/paired observations,
not numerical acceptance. Always retain exact numerical labels and scale notes;
colors and bar lengths do not imply a pass or a model ranking. Existing evidence
can be reused when its pinned hashes match; the branch import Guide records the
available archives and replay limits.

## Record a repair

Organize `data/repairs.json` as library groups → independent fixes → verification records. Declare each group in `library_groups` with a stable `id`, `title` and `components` (one library or an interacting combination). An independent repair sets `library_group_id`; `affected_library` identifies where the defect is located. A verification follow-up sets `related_repair_id` to the independent repair and inherits its group. `verification_title` labels that round of verification.

Append an entry to `repairs` for each experiment. Give each repair a stable ID and date, model identity, actual hardware/dependency versions, trigger, cause, change, verification stages and evidence links. A later verification gets a new entry linked to the same repair; a separate defect gets a new independent repair entry. Preserve earlier measurements, IDs, links and evidence. The page counts independent fixes separately from verification rounds. Use `status: "validated"` for completed verification and keep `validation.acceptance` separate (`null` means pending user judgment).

Metrics carry before/after values and units. State observed/expected coverage and verification scope. Keep unattributed residual differences in `validation.remaining`. Publish concise evidence under `evidence/`; retain full execution artifacts with their private cases. Add libraries, environments and models through data entries without changing the page.

## Validate and publish

Run `python3 -B .github/validate_history.py` and `python3 -B -m unittest discover -s tests -p 'test_*history.py' -v`. The optional `node tests/browser_history.cjs` regression requires Playwright and Chromium; `PRECISION_BROWSER_EXECUTABLE` selects an existing browser executable. It checks both tabs, filters, comparison values, permalinks, downloads, request races and mobile overflow. It starts and stops its own local preview server.

The Pages workflow validates both data files, uploads only `site/` and deploys on changes to `main`.

With Node.js available, run `python3 -B -m unittest discover -s tests -p 'test_evaluation_charts.py' -v`
for signed, zero, extreme-value and metric-domain geometry checks. The browser
regression checks every recorded comparison's chart values, side labels,
coverage and linear geometry against the archived measurements.
