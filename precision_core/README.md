# Shared execution and evidence

`precision_core` owns task-independent execution, reusable resource identities and
portable evidence. It does not own numerical acceptance, model workloads, public
community records or agent research policy. Import modules on demand:

```python
from precision_core.record_run import record_run
from precision_core.workflow import run_workflow, inspect_workflow
from precision_core.resource_catalog import find, register
from precision_core.runtime_probe import probe_runtime
from precision_core.evidence_bundle import fetch, pack, verify
```

The original contracts live in the owning module docstrings. Modules use the
standard library; native probes execute only the task's declared checks. Numerical
comparators, collection adapters and comparison-specific replay remain in
[precision-compare](../skills/precision-compare/SKILL.md).

From a checkout, use `python -B -m precision_core.record_run --help` or
`python -B -m precision_core.workflow --help`. The old Compare script paths remain
compatibility entrypoints and normal legacy imports share the same implementation.
RSI imports this package directly and does not add Compare scripts to `sys.path`.

## Portable skills

Export rather than copying Compare alone:

```sh
python -B -m precision_maintenance install --output /path/to/client/skills
```

The exporter copies skills and places one bundled copy of this package inside
Compare's scripts directory. No repository checkout, RSI package, editable install
or `PYTHONPATH` is needed at runtime. Existing destinations are rejected; export to
a fresh directory when updating, inspect it, then replace the installed version.
The legacy scripts resolve their package without changing the process import path.
Tool discovery reads canonical source statically, including in the exported skill.

The workflow engine fingerprints its canonical implementation files. This
migration therefore invalidates old execution-engine reuse identities. Historical
runs remain untouched; inspect workflows before deciding whether an affected
validation needs a new run. There is no automatic migration that marks old results
as verified under changed tooling.
