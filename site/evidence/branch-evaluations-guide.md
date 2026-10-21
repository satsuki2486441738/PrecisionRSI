# Validation branch evaluation import

This guide accompanies the evaluation records imported on 2026-10-22 from
`validation/gpt-6-cold-skill-20261019`. It describes a publication of existing
measurements, not a new accelerator run. The website retains its existing
Evaluation and Repair layouts and filters. Existing history records are preserved.

## Immutable sources

All imported source links are pinned to commit
`70bc3a2d0d04a6f2b688f557d022ed98e31eee01` in
[`satsuki2486441738/precision-skills`](https://github.com/satsuki2486441738/precision-skills/tree/70bc3a2d0d04a6f2b688f557d022ed98e31eee01).
The archive revision identifies the published evidence snapshot; it does not
replace the framework, model or tested skill revisions recorded inside it.

- [Latest completion matrix and Guide](https://github.com/satsuki2486441738/precision-skills/blob/70bc3a2d0d04a6f2b688f557d022ed98e31eee01/tests/results/gpt-6/compare-remainingcomplete-20261020T140716Z/GUIDE.md): 16 libraries, 17 task lanes, 34 canonical cases with independent references and 138 core comparison groups. Of these, 136 groups have public numerical payloads; two pretrained Bridge full-tensor groups have source-only evidence.
- [Earlier local framework matrix](https://github.com/satsuki2486441738/precision-skills/blob/70bc3a2d0d04a6f2b688f557d022ed98e31eee01/tests/results/gpt-6/compare-frameworkmatrix-20261019T124628Z/GUIDE.md): 15 selected native component/model cases. These are separate experiments, not additional completions of the 17-lane checklist. In particular, the local RL cases did not execute a framework training step, framework optimizer or rollout.
- [Independent cold-context inference case](https://github.com/satsuki2486441738/precision-skills/blob/70bc3a2d0d04a6f2b688f557d022ed98e31eee01/tests/results/gpt-6-luna/compare-luna-native-20261019T140106Z/results.json): one bounded llama.cpp CPU experiment, with later metadata corrections disclosed.
- [Tool hardening and retained-evidence validation](https://github.com/satsuki2486441738/precision-skills/blob/70bc3a2d0d04a6f2b688f557d022ed98e31eee01/tests/results/gpt-6/compare-skillhardening-20261020T180218Z/GUIDE.md): capture, archive and replay safeguards; no new native model cases.

[The import index](branch-evaluations-index.json) maps website IDs to source case
IDs, records the imported cohort counts and hashes the source summaries.

## Reading the records

Each experiment declares its compared sides, actual hardware, libraries,
settings, aligned conditions and known differences. Multiple precision modes or
engines can share one experiment when they use the same framework/model pair;
their comparisons remain separately selectable. Dates are the original
experiment dates, not the import date. Missing source observations are labelled
as unrecorded rather than inferred from a requested configuration.

The three imported cohorts cover **50 selected paired native cases**: 34 in the
latest completion matrix, 15 earlier local cases, and one independent cold-context case. They are not
50 independent input datasets or 50 new runs. The website experiment count can
be smaller because one experiment can contain multiple cases. Its comparison
count counts selectable peer comparisons; the completion matrix's 138 counts
core metric groups. Neither count includes validators or capture retries.
`archive.replay_cases` counts the retained numerical comparison groups that can
be recomputed from the public payload for that entry; it is not a native-job
count. Where original arrays are omitted, published reports remain inspectable
but the missing comparisons are not advertised as portable replay.

All imported records retain `acceptance: null`. `completed` means that the
declared collection and comparison finished. It does not mean a precision
tolerance was met. Metrics preserve recorded numerical values and units;
coverage counts paired observations, not equal arrays. A sampled gradient or
update is not described as a full-model comparison. Differences in hardware,
storage dtype, forward shape, cache handling or source adaptation remain visible.
Absolute differences of archived side values are scalar arithmetic summaries.
The difference of two side L2 norms is labelled separately from the L2 norm of
the elementwise difference; these quantities must not be confused.

## Completion boundaries and exclusions

- The original Ascend instance exposed 16 logical devices. These experiments
  establish bounded workflows, not 16-device scaling. TensorRT-LLM ran its
  selected CUDA backend on the local RTX 5060; it has no NPU generation result.
- Successful community forks, API migrations or compatibility patches do not
  convert the historical untouched NeMo NVIDIA/NCCL, Bridge 0.6.2/TE/CUDA or
  TensorRT 0.20.0 ARM/Ascend failures into successes. Their original outcomes
  remain in the pinned completion Guide and earlier matrix.
- Slime's native postclip arrays and original gradient dtype were not captured.
  Bridge's pretrained full gradient/update arrays remain private. TensorRT's
  converted weights and cache dtype were not directly measured. Per-entry notes
  retain these and the other source-specific limits.
- The earlier four-H100 TRL run is a console-observed native smoke without a
  matched numerical reference or downloaded per-rank evidence. It is linked in
  the earlier matrix and excluded from the paired evaluation count.
- Synthetic artifact examples, runtime import/entrypoint probes, failed setup
  attempts, repeated captures and the NeMo tiny zero-reward CLI smoke do not add
  canonical model precision cases. Tool regression and replay checks remain in
  their original maintenance Guide.

## Verification and reuse

The data import itself changes only evaluation data and concise public evidence. Large
arrays, checkpoints and existing archives retain their original locations and
bytes. Use each record's pinned evidence links and archive prefix to find the
specific manifests, reports, source versions and replay instructions. The later
portable conversion of the cold-context case preserves the selected raw arrays and records its
small floating-point replay differences; it does not rerun the model.

Before publishing, validate both histories with
`python -B .github/validate_history.py`, run the repository tests and metadata/link
checks, and exercise both tabs, filters, comparison selection, deep links and
JSON download with the existing browser regression. Preserve the original
Repair data and site presentation files during this data import.

## Evaluation charts (2026-10-22)

The subsequent chart update reuses the retained evidence. All 55 pinned source
files were checked against their Git blob identities, byte counts and SHA256
hashes. The 39 ZIP files contain 4,707 members whose CRC checks passed. These
records were already present locally, so no duplicate download or native model
execution was needed.
An additional semantic audit compared 157 metrics and their side values across
six experiments and 17 comparison scopes with the corresponding ZIP reports;
all values matched exactly.

The Evaluation tab uses the same gray/green bars, backgrounds, borders and
typography as Repair. Where both side values were recorded, the chart labels
the actual compared configurations and uses a common linear domain including
zero. Negative values retain their signs. The central comparison result remains
separate from the two side values: for example, a difference between side L2
norms is not the L2 norm of the elementwise difference.

Nonnegative error summaries share a linear scale only with the same unit in the
selected comparison. JS divergence uses its own group, separate from logprob
errors even when both are recorded in nats. Cosine charts use [-1, 1], and agreement/overlap charts use
[0, 1]. An unrecognized or out-of-range metric remains numerical rather than
receiving a guessed domain. Exact labels remain available even when a tiny
value has no visible bar width. All-zero data is not given an artificial
nonzero magnitude. Coverage bars represent observed/expected collection counts.
Neither colors, display scales nor full coverage establish numerical acceptance.

Charts refresh when the comparison selection changes. The update preserves the
evaluation data, original reports, archive hashes and acceptance decisions.
Browser verification covered all 119 selectable comparisons and 1,326 metrics,
including displayed values, side ordering, linear geometry, coverage and mobile
overflow. Geometry regressions additionally cover negative values, zero-only
domains, extreme finite numbers, tiny values and separation of JS/error scales.
