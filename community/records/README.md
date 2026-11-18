# Community records

This package validates and indexes community evaluations, issues, solutions and
verification evidence. It runs locally over versioned JSON.

```sh
python -B -m community.records validate
python -B -m community.records search --hardware H200 --model Qwen3.5 --model-size 9B --library PEFT
python -B -m community.records case --id case-qwen35-weighted-lora-bias
python -B -m community.records verification --id verification-qwen35-weighted-lora-bias
python -B -m community.records inventory
```

Commands accept `--site DIRECTORY`. Read commands accept `--output NEW_JSON`.
For Agent use, search by hardware, model and framework, then use the returned case
ID or the solution's `verification_ids`. `case` includes the original settings,
metrics and relations. `verification` returns the original values with their
source references and recorded acceptance. These commands read the public catalog
directly and do not depend on a browser or a running site.

## Record organization

| Information | Owner |
|---|---|
| Model architecture, parameter size, checkpoint and revision | Evaluation or repair `model` |
| Hardware, libraries, settings | Evaluation sides or repair environment |
| Framework under test | Evaluation side `frameworks`, repair `affected_library`, or `data/frameworks.json` for historical observations |
| Measurements, coverage, run status, acceptance | Original evaluation or repair verification |
| Cases, issues, solutions and their relationships | `data/relations.json` |
| Public titles, summaries and affected workflows | Record `content`, or `data/content.json` for historical records |
| Reproduction, implementation details and evidence | Referenced source records and evidence files |
| Search projection | Generated `data/search-index.json` |

Keep each fact in its owning field. A title states the topic; a summary states
the observation or proposed change. Put versions, commits, configurations and
verification outcomes in their structured fields.
Versions contain version identifiers; local patch and backport descriptions
belong to solution records. Source revisions have their own commit fields.
Record a specific unresolved condition where it affects use.
General reminders about scope, correctness or
hardware attribution do not belong in every record.

### Public content

New evaluations and repairs can include `content`; lifecycle entities can include
`details.content`. It contains a required localized `title`, an optional localized
`summary`, and optional `workflows` describing affected inference/SFT/RL usage:

```json
{
  "title": {
    "zh": "LoRA 合并后的输出不一致",
    "en": "LoRA output changes after merging"
  },
  "summary": {
    "zh": "合并适配器时遗漏偏置。",
    "en": "Adapter merging omits the bias."
  }
}
```

Each localized field supports `zh` and `en`; one language is sufficient. Titles
are at most 120 characters and summaries at most 400 characters per language.
For historical records, add the same object to `data/content.json`:

```json
{
  "schema_version": 1,
  "records": [
    {
      "kind": "issue",
      "id": "issue-qwen35-weighted-lora-bias",
      "content": {"title": {"zh": "LoRA 合并后的输出不一致"}}
    }
  ]
}
```

Typed identities must exist and have one content definition. This keeps editorial
changes separate from historical measurements. `Catalog.content(kind, id)` returns
the description; `Catalog.get(kind, id)` returns the original record, locator,
and public content when present.
The website and search compiler consume these same fields.

## Demand retrieval

```python
from community.records.catalog import Catalog
catalog = Catalog('community/site')
result = catalog.search({'hardware': '910_9382', 'model': 'Qwen3', 'model_size': '0.6B',
                         'library': 'ms_swift', 'task': 'rl'}, limit=10)
```

Hardware, model and framework identify the target. In `search`, `library` names
the framework under test or repair target on that side. The full dependency
stack remains in `libraries`. The legacy `cases` command can still look up
packages in the environment. Use a model group such as
`Qwen3 Dense` or `Qwen3 MoE` to browse, or an exact recorded checkpoint name
for a recommendation. The API also accepts `model_size`, task and library version.
The search normalizes indexed aliases, including H20Z → H200 and
910_9382 → Ascend 910C. Device count and execution settings remain with the
configuration. Unknown names remain searchable as entered.

On new evaluation sides, use `frameworks` to list the libraries whose execution
is being evaluated. Each must also appear in that side's `libraries` list.
Reference-only methods can omit `frameworks`. For historical evaluations,
`data/frameworks.json` maps evaluation IDs to framework names and their exact
side IDs. Use `null` when all sides evaluate the framework.
For historical investigations without an `affected_library`, the `repairs` map
lists the frameworks tested in the recorded environment without assigning cause.

The response includes normalized input, matching cases, issues, solutions,
recommendations, version alternatives and source identities. No match returns an
empty result; the Agent continues its normal evaluation/repair task. Existing
observations remain available when recommendation support is incomplete.
`recommendation_gaps` gives the missing requirements as structured codes.

`alternatives` lists supported versions of the same hardware/model/size/library/task
combination. `--allow-alternatives` permits selecting one. The default query
returns exact matches. `--query`, `--limit` and `--offset` provide keyword search
and pagination; `--include-workflows` includes Agent workflow benchmarks.

Recommendation eligibility uses recorded versions, measured coverage and
verification links. Select an exact checkpoint to retrieve recommendations. Each returned
configuration retains its checkpoint and revision. Solutions expose their
`issue_ids` and `verification_ids`.
The original numerical acceptance field retains its recorded value.

For new records, include `model.architecture` and `model.size` when known. Use the
architecture name separately from `model.name` and `model.revision`. Use `size`
for a parameter scale such as `0.5B`, or a decimal count when known precisely.
Use `tiny` for custom miniature fixtures; they share a size bucket within each
architecture, with each seed and actual shape preserved in the source record.

The standalone [search tool](../../tools/community_search.py) uses the same
metadata and is bundled with exported Compare skills. It accepts `--request JSON`,
`--index FILE_OR_HTTPS_URL` and an optional `--sha256 HASH`.

```sh
python -B -m scripts export-search --refresh
python -B -m scripts export-search --check
```

Regenerate the index after record or content changes. It pins the source catalogs
and content file. Observation dates describe the experiments; editorial updates
do not change them.

## Relationships

Lifecycle entities have `id`, `kind`, `scope`, `evidence` and optional `details`.
Kinds are `case`, `investigation`, `issue`, `candidate`, `verification`, `workload`.
`candidate` is the storage name for a solution. Entity IDs are stable slugs.

Every details object has `title` and `sources`. `content` and `limitations` are
optional. Add `limitations` for concrete technical conditions that affect use;
omit it when there is nothing additional to record.

| Kind | Additional fields |
|---|---|
| case | `request`: hardware, model, framework, task, goal; optional `category` |
| issue | `classification`: confirmed-defect/numerical-discrepancy/unconfirmed-hypothesis |
| candidate | `solution_type`, `target`, `applicability`, `evidence_level`, `decision` |
| verification | `method`, `outcome`, `acceptance` source reference |
| workload / investigation | No additional fields |

Case categories are `precision-evaluation`, `repair-investigation` and
`workflow-benchmark`. Solution types are patch, backport, upgrade, configuration
and numerical-policy. Targets carry library, version and commit. Evidence levels
are runtime-tested-in-scope and source-inspected. Decisions are
selected-for-further-study, not-selected and not-recorded. Verification methods
are native-runtime and retained-measurements; outcomes are scoped-checks-passed,
partial, mixed-results and not-adopted.

Links have `id`, `relation`, `from`, `to`, `scope` and `evidence`.

| Relation | From → To |
|---|---|
| investigates | investigation → comparison |
| confirms | investigation → issue |
| addresses | candidate → issue |
| verifies | verification → candidate |
| uses_workload | verification → workload |
| documents | issue → repair |
| measures | verification → comparison |
| derived_from | measurement → comparison |
| includes | case → member record |

A case contains its associated issue, solutions and verification workloads.
Multiple solutions can address one issue. Each verification links to a candidate
and an original workload. Validation checks endpoint types, required members,
duplicate links and evidence hashes.

Evidence items use `{file, sha256}` relative to the site. Structured sources use
`{file, sha256, pointer}` for frozen JSON evidence, or `{kind, id, pointer}` for an
existing comparison, repair or measurement. Pointers follow RFC 6901. Acceptance
references name an `acceptance` or `numerical_acceptance` field.

`case_bundle(id)` returns members and links. `verification_bundle(id)` returns
the verification and its resolved sources. `resolve_source(source)` reads the
referenced value. `index()`, `get()` and `related()` expose typed identities and
declared links. `find_cases()` and `find_comparisons()` retain exact historical
label matching; target-facing retrieval uses `search()`.

## Add or update a record

Write a short description and place the environment, measurements and evidence
in their structured fields. Link an issue to its observations, each solution to
the issue, and each verification to its solution and workload. Preserve original
IDs, values and evidence when adding a follow-up. Update public wording through
content metadata.

```sh
python -B -m scripts export-search --refresh
python -B -m community.records validate
python -B -m scripts validate-immutability --base origin/main
```

See the [submission writing guide](../../skills/precision-compare/references/community-records.md)
for Agent-authored summaries and the [publication policy](../../docs/publication.md)
for repository scope. The [reviewed inventory](../../reports/community-record-inventory-20261117.md)
and [association ledger](../site/evidence/community-record-associations-20261117.json)
retain the historical mapping. The [H200 naming correction](../site/evidence/hardware-name-correction-H200-20261117.json)
and [token-mean clarification](../site/evidence/token-mean-scope-clarification-20261117.md)
record the existing corrections.
