# Public repository scope

Publish reusable code, current guides and concise final results.
Developer regression suites remain local in the Git-ignored `tests/` directory.
Current guides live in `docs/`, final reports and figures in `reports/`;
community pages, public data and record tools are grouped under `community/`.
`.github/` contains GitHub platform configuration only.
A result identifies hardware, model, software revisions, settings, measured scope,
metrics, limitations and reproducibility or immutable evidence references.
Completion and numerical acceptance have separate fields.

Write public titles and summaries in record `content`; historical descriptions
use `community/site/data/content.json`. Describe the actual observation or change
in plain language. Store versions, settings, status and evidence in their owning
fields. Add specific tradeoffs or unresolved conditions when relevant; omit stock
scope/correctness disclaimers. See the [record format](../community/records/README.md)
and [writing guide](../skills/precision-compare/references/community-records.md).

## Keep public

- Framework and skill code, minimum reproducers, patches and workflow validators.
- Current architecture, usage and maintenance guides.
- Final reports, configuration-scoped metrics, SVG figures and evidence hashes.
- Published records and the evidence they reference, including relevant failures
  and unresolved gaps. A later result does not rewrite a historical observation.

## Keep local

- Developer regression and browser test suites in `tests/`; retain permanent cases locally.
- Optional RSI runner source in `research/`, its experiment state, candidates and full traces.
  Public dashboard snapshots may remain when they contain reviewed summaries only.

- Execution logs, handoffs, completed one-off plans and exploratory candidates.
- Raw tensors, checkpoints, full responses, traces and historical delivery ZIPs.
- Duplicate metric snapshots and generated PNG/PDF copies when the public SVG
  and its underlying final measurements already provide the same result.

Use one canonical metrics file per published result. Reports link to it instead
of copying the same JSON into each report directory. Keep aggregate CSV exports
when they are a documented user interface. Public snapshots are retained when
pages or immutable source references consume them; file size alone does not
make a source redundant.

Before removing a file, inspect code, page and record references; retain a local
byte-verified backup and verify the replacement source. Do not drop numerical
coverage, failure states or caveats while shortening the presentation. Validate
links, public history and the affected tools after cleanup. Private cases retain
full experiment evidence; preserve local regression coverage when changing a dependency.

Cleanup changes the current branch only. Previously committed material remains
in Git history; this policy does not authorize rewriting published history.
