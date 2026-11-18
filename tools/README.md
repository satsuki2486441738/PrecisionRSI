# Shared Skill tools

These are the common tools called by the skills: command execution, reusable
resource identities and portable evidence. This directory is not a separate
product or a required service. It does not own numerical acceptance, model workloads, public
community records or agent research policy. Import modules on demand:

```python
from tools.record_run import record_run
from tools.workflow import run_workflow, inspect_workflow
from tools.resource_catalog import find, register
from tools.runtime_probe import probe_runtime
from tools.evidence_bundle import fetch, pack, verify
from tools.community_search import load_index, search
```

The original contracts live in the owning module docstrings. Modules use the
standard library; native probes execute only the task's declared checks. Numerical
comparators, collection adapters and comparison-specific replay remain in
[precision-compare](../skills/precision-compare/SKILL.md).

`community_search` is a read-only metadata client shared by Compare, Adapt and
Repair. Community-owned compilation binds its index to public catalogs and
original-workload evidence; the tool queries that snapshot without importing the
community package or starting a model. It returns matching records and separate
version alternatives. An empty match supplies no additional context; continue the
normal evaluation or repair workflow. Public descriptions come from the same
content fields as the website.
For original metrics and source values, query the
[community records interface](../community/records/README.md) by a returned case
or verification ID in a checkout of the public catalog.

From a checkout, use `python -B -m tools.record_run --help` or
`python -B -m tools.workflow --help`. The old Compare script paths remain
compatibility entrypoints and normal legacy imports share the same implementation.
RSI imports this package directly and does not add Compare scripts to `sys.path`.

## Portable skills

Export rather than copying Compare alone:

```sh
python -B -m scripts install --output /path/to/client/skills
```

The exporter copies skills and places one bundled copy of this package inside
Compare's scripts directory. No repository checkout, RSI package, editable install
or `PYTHONPATH` is needed at runtime. Existing destinations are rejected; export to
a fresh directory when updating, inspect it, then replace the installed version.
The legacy scripts resolve their package without changing the process import path.
Tool discovery reads canonical source statically, including in the exported skill.
The export includes a dated public search metadata snapshot for offline retrieval.
Use the search tool's `--index` and optional `--sha256` to select a different
reviewed snapshot explicitly; queries return the loaded identity and source pins.

The workflow engine fingerprints its canonical implementation files. This
migration therefore invalidates old execution-engine reuse identities. Historical
runs remain untouched; inspect workflows before deciding whether an affected
validation needs a new run. There is no automatic migration that marks old results
as verified under changed tooling.

## Which tools belong here?

Keep a tool here when multiple skills or the optional research runner use it.
Comparison-specific array/distribution arithmetic and collectors stay in
`skills/precision-compare/scripts/`. Skill entrypoints resolve these shared tools;
there is one maintained implementation, not a separate copy for each skill.
The exporter bundles the required implementation for standalone installation.

Repository imports use `tools.<module>`. Maintainer commands use `python -m scripts`;
optional local research experiments are not distributed in the public repository. Neither maintenance nor
research code is required in an exported skill installation. Existing private
runs and version receipts retain their original paths and source identities;
changed tools require fresh affected verification, not rewritten old receipts.
