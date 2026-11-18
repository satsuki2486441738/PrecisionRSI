# Community website

Preview from the repository root:

```sh
python3 -m http.server 8000 --directory community/site
```

## Pages

| Route | Content |
|---|---|
| `index.html` | Comparison board, platform views and repair results |
| `index.html#/cases` | Target selection, library combinations, issues and solutions |
| `index.html#/report/<case-id>` | Measured report for one comparison or verification linked from the case library |
| `index.html#/list` | Historical measurement list and existing record links |
| `archive.html` | Evaluation, repair and measurement archives |
| `rsi.html` | RSI iteration summaries |

Opening the case library without filters shows a measured example. Reset clears
the filters; links with a target or a case open that selection directly.

## Shared data

`data/evaluations.json` and `data/repairs.json` own experiment configurations and
measurements. `data/relations.json` connects cases, issues, solutions and
verifications. Public titles and summaries live in record `content` or
`data/content.json`. The board, case library, archive and Agent retrieval consume
these descriptions. See the [record model](../records/README.md).

`board-data.js` derives the board's metrics and groups at page load.
`community.records.search` compiles `data/search-index.json` for both `search.js`
and the standalone Skill search tool. The generated index pins all contributing
catalogs, including content and historical framework annotations. Refresh it
after changes to those sources:

```sh
python -B -m scripts export-search --refresh
python -B -m community.records validate
```

Titles describe a task or problem. Summaries explain its effect or solution in
plain language. Versions, runtime settings, verification outcomes and evidence
are separate fields. Implementation detail and historical notes are available in
the detail views. Missing optional summaries render as empty rather than a stock
warning. Editorial updates use typed record IDs; frontend JavaScript does not
contain per-case title overrides.

## Configuration and issue discovery

The left panel selects hardware, model, framework and an optional framework
version. Framework choices are grouped by recorded workload: training (SFT),
inference and RL. A framework can appear in several groups; selecting it also
selects that workload and scopes the available versions. Each group shows up to
three rows. Its overflow button opens the shared framework drawer at that group.
The drawer offers search and an unscoped list for records without a workload
label. Framework options come from the measured sides and repair targets.
Dependencies and their versions stay in the configuration details.
Inline options come from actual catalog values, ordered by matching record count.
The “…” drawer exposes the full inventory and custom targets. The example button
selects an eligible recorded configuration from the current index.

The right panel displays a recommended configuration and its issues. Opening an
issue shows its solutions, verification outcomes and tested versions. Evaluation
cases show a short result summary. Use the [community records interface](../records/README.md)
for machine retrieval of exact settings, metrics and source evidence.

A query without matching records returns an empty result. The Agent can continue
its normal comparison/repair workflow. Records with insufficient recommendation
support remain available in the report and archive. Other supported library
versions use a separate version-selection action.

Recommendation support is compiled from pinned dependencies, measured coverage,
recorded acceptance and links between the solution, verification and original
workload. Resolution also requires a link from the solution to the selected issue.
Partial, rejected and source-inspected candidates retain their own states.

Hardware aliases group H20Z under H200 and 910_9382 under Ascend 910C. Device
counts, topology and raw settings remain attached to the actual configuration.
Unknown hardware and model names retain their identities.

## Maintenance

`cases.js` and `cases.css` implement the case route. `search.js` queries the shared
index and resolves localized content. `evaluation-presentation.js` and
`repair-presentation.js` support archive clients; current records supply content
metadata, while older standalone callers retain their formatting fallback.
`differences.js` displays measurement views and their original source references.

Source catalogs are checked against the search index when loading the case route.
Use localhost or HTTPS for browser hash APIs. Query parameters and case/issue deep
links survive reloads.
The site is static; submissions are prepared as records and reviewed in Git.

The [publication guide](../../docs/publication.md) describes public artifacts.
The [reviewed inventory](../../reports/community-record-inventory-20261117.md),
[association ledger](evidence/community-record-associations-20261117.json),
[H200 correction](evidence/hardware-name-correction-H200-20261117.json) and
[token-mean clarification](evidence/token-mean-scope-clarification-20261117.md)
retain the historical record mapping.
