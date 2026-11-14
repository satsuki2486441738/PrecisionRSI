# Precision history website

GitHub Pages serves this directory. Preview from the repository root with `python3 -m http.server 8000 --directory site`, then open `http://localhost:8000`.

## Pages

| Path | What it is |
| --- | --- |
| `index.html` | NumericForge — the comparison board: exploration swarm, question groups, detail list, platform compare and repair figure. |
| `archive.html` | The repair / evaluation / differences dashboard this directory served previously. Kept at its own URL; the board does not link to it. |
| `rsi.html` | RSI iteration console, driven by `data/rsi.json`. |

`index.html` derives everything it shows from `data/evaluations.json` and
`data/repairs.json` at load time through `board-data.js`; nothing is baked into
the page, so adding a record to those files updates the board with no build
step. Because it reads sibling files, it needs a static server — opening it
through `file://` is blocked by the browser and the page says so.

`board-data.js` normalises free-text hardware strings into platforms, maps
bilingual metric labels onto one canonical id per quantity, and classifies each
comparison by subject and by the single axis it isolates, reporting what stays
unclassified rather than forcing a bucket. Readings that disagree in direction
are never ranked together: cosine similarity and Top-1 agreement are ordered by
deviation from agreement, and no total is computed across groups whose units
differ.

## Survey differences

The **Differences** tab (`#differences`) of `archive.html` collects the five `survey/*` deliveries in
`data/differences.json`, with the original public JSON payloads retained separately
in `data/differences-raw.json`. It groups measurement views by model, including a
separate operator-reference group. Keyword, comparison-type, model and platform
filters, paginated cards, per-metric charts, configuration details and immutable
source links keep the complete collection browsable. `#difference-<id>` opens the
matching card even when it is on a later page. Downloads preserve the full archive.

Direct GPU–NPU comparisons, within-platform native/reference comparisons,
within-platform configuration comparisons, operator references and diagnostic
views remain distinct. Shared checkpoints or inputs do not make differing
precision, parallelism, placement or software stacks hardware-only comparisons.
Counts describe recorded views, including overlapping aggregate/detail views,
not independent experiments or statistical samples. Collection completion does
not grant numerical acceptance; deferred task coverage remains explicit.

The collector pins every branch to a full commit SHA, verifies downloaded Git
blob hashes and the shared archive manifest, and deduplicates identical source
content. Every numeric metric and paired value retains its JSON Pointer into a
frozen public source. Private tensors, checkpoints and full execution stores are
not copied into the website. Metadata remains in the source payload; it is not
presented as a precision error. Charts preserve signed values and use a common
linear zero baseline within the displayed series. Different comparison types,
source paths and stages are kept separate; no model ranking or pooled error is
computed across heterogeneous protocols.

Regenerate the snapshot with the standard-library collector:

```sh
python3 -B .github/collect_differences.py --output site/data/differences.json --raw-output site/data/differences-raw.json
python3 -B .github/validate_differences.py
python3 -B -m unittest discover -s tests -p 'test_difference_history.py' -v
```

To collect a later delivery, review and update the collector's pinned `PINS` and
archive digest explicitly, then regenerate and verify the output. This is a
reproducible published snapshot, not a live polling service. The optional
`node tests/browser_differences.cjs` regression drives `archive.html` with Playwright and an installed
Chromium (`PRECISION_BROWSER_EXECUTABLE`) to check filters, pagination, deep links,
exact values, downloads, failure/retry behavior and mobile layout. The Pages
workflow validates the survey archive against its frozen sources before deployment.

## Record an evaluation

The **Evaluations** tab reads `data/evaluations.json` and groups experiments by inference, SFT and RL. Each experiment has a stable ID/date, model revision, common conditions, comparison variables, actual per-side hardware/libraries/settings, and an immutable archive revision. A comparison references two declared side IDs and owns its scope, numerical metrics and observed/expected coverage. An experiment can contain multiple steps or execution modes without becoming multiple history entries.

Metric `value` is the symmetric comparison result. Optional `side_values` records each compared side's value in the same unit; optional `coordinate` identifies the maximum error. Keep finite numbers at recorded precision. Equal-array counts are measured results; coverage counts collected/paired observations. Set `acceptance: null` until a numerical acceptance decision is recorded. `status: "completed"` means collection/comparison finished.

Append concise public evidence under `evidence/` and keep native arrays/checkpoints in the existing artifact store. Preserve recorded IDs, values and immutable provenance. The initial entries cover same-H200 BF16/FP16 inference, native cross-platform TRL LoRA SFT and same-platform checkpoint restoration with peer comparison. New frameworks and hardware require data entries, not UI changes.

Set `visibility: "archived"` to retain an experiment in the raw archive and JSON download while excluding it from cards, counts, filters and charts. Historical skill efficiency experiments are currently archived because they measured earlier skill revisions. Numerical precision comparisons remain visible.

Links use `#evaluations` and `#evaluation-<id>`. Existing repair/fix links still select the Repairs tab. Both tabs support keyword, library, model and hardware filters; evaluations also support task filtering and comparison selection. JSON downloads contain the selected archive.

Evaluation charts use the Repair tab's gray/green palette. Only two recorded,
unequal side values receive bars, with the actual side labels and a common
linear domain including zero. Negative values retain their signs; no relative
cutoff hides small differences. Equal pairs, scalar errors, cosine/JS summaries
and coverage counts remain numerical, without decorative bars. Keep exact labels;
colors and bar lengths do not imply acceptance. Existing evidence can be reused
when its pinned hashes match.

Display nested settings as labelled fields and groups rather than serialized
JSON. The recorded-conditions wrapper is expanded into model, inputs, numerics,
execution and scope. Put long arrays and revision/hash details behind disclosures,
retain empty/false/zero/unknown states, and keep original values in the JSON
download. Human-readable execution descriptions must preserve the recorded
meaning. Render all supplied text as text, never HTML.

Synthetic repair fixtures belong in Evaluations; reserve Repairs for established implementation defects. For skill A/B experiments, retain the shared context, loaded skill revision, execution order, raw per-response usage and pricing snapshot in evidence. Report preparation separately from evaluator work; publish failed or slower outcomes alongside successful ones.

## Record a repair

Organize `data/repairs.json` as library groups → independent fixes → verification records. Declare each group in `library_groups` with a stable `id`, `title` and `components` (one library or an interacting combination). An independent repair sets `library_group_id`; `affected_library` identifies where the defect is located. A verification follow-up sets `related_repair_id` to the independent repair and inherits its group. `verification_title` labels that round of verification.

Append an entry to `repairs` for each experiment. Give each repair a stable ID and date, model identity, actual hardware/dependency versions, trigger, cause, change, verification stages and evidence links. A later verification gets a new entry linked to the same repair; a separate defect gets a new independent repair entry. Preserve earlier measurements, IDs, links and evidence. The page counts independent fixes separately from verification rounds. Use `status: "validated"` for completed verification and keep `validation.acceptance` separate (`null` means pending user judgment).

Metrics carry before/after values and units. State observed/expected coverage and verification scope. Keep unattributed residual differences in `validation.remaining`. Publish concise evidence under `evidence/`; retain full execution artifacts with their private cases. Add libraries, environments and models through data entries without changing the page.

## Validate and publish

Run `python3 -B .github/validate_history.py` and `python3 -B -m unittest discover -s tests -p 'test_*history.py' -v`. The optional `node tests/browser_history.cjs` regression requires Playwright and Chromium; `PRECISION_BROWSER_EXECUTABLE` selects an existing browser executable. It checks both tabs, filters, comparison values, permalinks, downloads, request races and mobile overflow. It starts and stops its own local preview server.

The Pages workflow validates both data files, uploads only `site/` and deploys on changes to `main`.

With Node.js available, run `python3 -B -m unittest discover -s tests -p 'test_evaluation_charts.py' -v`
for signed, zero, extreme-value and selective-chart geometry checks. The browser
regression checks every recorded comparison's chart values, side labels,
coverage and linear geometry against the archived measurements.

## RSI dashboard

`rsi.html` presents runtime snapshots separately from precision histories. Select a run to inspect candidate components, acceptance, task correctness, tool calls and baseline/construction/evaluation usage. Accepted versions are scoped to each run; acceptance alone does not demonstrate reuse or efficiency. Unknown values remain unknown. Reused baseline receipts retain their historical cost and add zero baseline calls to the new run.

Refresh the public summary after a maintenance check:

```sh
python -B .github/rsi_snapshot.py --monitor projects/rsi-system/monitor.json --output site/data/rsi.json --refresh
```

Omit `--refresh` to export archived local evidence. The optional refresh reads the current remote run through the monitor's SSH declaration; it does not change the run or model service. The exporter returns allowlisted summaries, strips paths from diagnostics, and keeps full traces, task inputs, SSH settings and credentials private. The page polls the published JSON every minute and shows its observation time; publication is a snapshot, not a direct connection to the model server. Publish `site/data/rsi.json` through the existing Pages workflow.

Run exporter regressions with `python -B -m unittest discover -s tests -p 'test_rsi_dashboard.py' -v`. Optional page checks use `node tests/browser_rsi.cjs` with Playwright and an installed Chromium.
