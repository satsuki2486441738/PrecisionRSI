# Public record model

This package owns public comparison/repair validation, measurement-view validation
and a read-only record index. It has no dependency on `precision_rsi`, agent
harnesses, model runtimes or execution tools. The community remains versioned JSON
and static pages; this package is not an upload service or database.

```sh
python -B -m precision_community validate
python -B -m precision_community index --output /tmp/public-record-index.json
```

Both commands accept `--site DIRECTORY`. Index output must be new. Validation
checks data structure, local evidence and declared relationships; it does not
establish numerical acceptance or independently verify causal claims.

```python
from precision_community.catalog import Catalog
catalog = Catalog('site')
records = catalog.find_comparisons(hardware='Ascend 910C',
                                  library={'name': 'torch_npu', 'version': '2.9.0.post2'},
                                  task='rl')
```

Hardware and library constraints match the **same recorded side**. Matching is
exact, with no source-name normalization. `get(kind, id)` returns a detached source
record, catalog SHA-256 and JSON Pointer. `index()` lists typed identities and
locators. `related(kind, id)` returns declared incident links only. Measurement
views remain measurement views rather than new independent experiments.

## Explicit relationships

[relations.json](../site/data/relations.json) is an additive layer over existing
catalogs. It starts empty: historical similarity is not a causal relationship.
Existing comparisons, repairs and measurement views use typed catalog IDs. New
entities have exactly `id`, `kind`, `scope`, `evidence`; kinds are `investigation`,
`issue`, `candidate`, `verification`, `workload`. Entity IDs are globally unique
slugs within the entity collection.

Links have exactly `id`, `relation`, `from`, `to`, `scope`, `evidence`. Endpoints are
`{"kind": "comparison", "id": "existing-evaluation-id"}`-shaped objects. Link IDs
are unique slugs. Allowed directions are:

| Relation | From | To |
|---|---|---|
| investigates | investigation | comparison |
| confirms | investigation | issue |
| addresses | candidate | issue |
| verifies | verification | candidate |
| uses_workload | verification | workload |
| documents | issue | repair |
| measures | verification | comparison |
| derived_from | measurement | comparison |

Every entity and link requires a nonempty scope and evidence list. An evidence
item is `{"file": "evidence/versioned-report.json", "sha256": "<64 hex digits>"}`
relative to site. Evidence must exist inside site and match its hash. Prefer
immutable, versioned evidence files to mutable aggregate catalogs. A verification
must link to both a candidate and an original workload. Duplicate edges, dangling
IDs, wrong endpoint kinds and incomplete verification bindings are rejected.

The relation file is protected by the same append-only guard as published
catalogs. Corrections require new identities and explicit scope; old claims and
failed investigations stay recorded. A relation label is a reviewed assertion,
not a conclusion produced by the validator. Human review must check that its
bounded evidence supports that assertion; links never change a record's status or
null acceptance. Full private runs and arrays remain outside public records.
