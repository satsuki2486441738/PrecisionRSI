# Environment and resource evidence

## Shared implementation versions

Identify libraries used by both sides for the actual loading, input processing, computation, or scoring path. Examples include Torch, Transformers, and the tokenizer implementation. Match their upstream release versions, including patch versions. For source installations, align commits and relevant local implementation changes. Distinct libraries that are themselves the subject of a comparison retain their separate identities; their shared dependencies follow the same version rule.

Keep CUDA, CANN, torch-npu, and other platform extensions compatible with their respective stacks. Record build strings such as `+cu126` and `+cpu` separately from the common upstream release. Include customized changes affecting the shared implementation in the version review. A Torch `+cpu` build can execute on Ascend through torch-npu registration; assess the required APIs and the actual backend rather than the suffix alone.

Record the shared package list, target releases, observed versions, and review result in the case. A first single-side baseline establishes target releases for the other side. Inventory packages relevant to the path; distinguish installed support packages from modules actually imported.

For a different-version proposal, present the libraries and exact versions, the reason, its effect on interpretation, and the same-version alternative or alignment obstacle. Obtain explicit user approval for that version pair and comparison scope before the corresponding collection or comparison. Reuse approval within its original scope; new differences need a new decision. Permission to reuse an environment or execute a device task does not itself authorize a version difference. Reference the approval from the manifest's `side_config` and explain the stack difference in the report.

Version admission belongs to the collection/comparison coordinator. The schema-v2 numerical core validates its documented fields; it does not enforce this package policy automatically.

## Declarations and execution association

An environment declaration describes the intended dependencies. A lockfile resolves versions and installation sources. Runtime records identify the environment and implementation that produced an observation. Use these records together to establish the versions for a run.

Accept either of these evidence paths when it identifies the required shared releases:

| Evidence path | Association with the run |
|---|---|
| Runtime inventory | The collector records its interpreter, package releases, runtime build strings, and relevant loaded module locations with the run and rank. |
| Environment preparation and launch records | A successful locked installation or immutable image record identifies the environment; the launch identifies that same interpreter/environment or image; available records establish that relevant dependencies were unchanged through execution. |

A missing dedicated version JSON is a lookup gap, not an automatic rerun requirement. First examine the existing lockfile, installation/sync result, launcher, interpreter prefix, image identity, and relevant environment change records. A lockfile alone establishes a declaration; add the preparation and launch association needed for that particular run. A filesystem path alone does not establish that its contents stayed unchanged.

If the original environment is still available, inspect it using the same interpreter and record the inspection time. Attribute its versions to a historical run only when the unchanged-environment association is established. Otherwise describe the inventory as current and identify exactly which historical versions remain unresolved. Do not insert current observations into old execution records.

Prefer collecting the association once over requesting repeated evidence. For new runs, save a compact inventory at startup, before model loading. Include:

- The run, rank, timestamp, interpreter executable/prefix, and environment or image identifier where available.
- Participating distribution releases, runtime version strings when exposed, and relevant imported module locations. Record an unavailable module version as unavailable; package metadata remains useful.
- The lockfile or preparation record used, and the launch association. Record a commit or relevant patch for source installations.
- Platform extensions, driver/runtime identity, and effective numerical settings separately from common releases.

Homogeneous ranks may share one inventory when each rank is associated with that environment and the execution system provides the relevant homogeneity evidence. Otherwise capture the required versions per rank. Keep inventory generation lightweight and avoid importing unused frameworks just to list versions.

Routine checks use package metadata, environment preparation, and load locations. Full source hashes are reserved for a concrete integrity concern or suspected implementation change. Unknown historical versions call for bounded investigation and a minimal collection proposal when necessary; they are distinct from an observed version mismatch.

## Project environment layout

Use the project's existing environment manager, including existing containers:

```text
projects/<project>/environments/<gpu|npu>/<framework-version>/
  <environment declaration and lockfile>
  setting.yaml
```

For example, a Pixi environment may use `pixi.toml` or `pyproject.toml` with `pixi.lock`. `setting.yaml` records selected builds, Python, devices, driver/runtime, numerical switches, and capability evidence. Keep the environment installation and model weights at their own resource locations.

Each side's manifest references its declaration through `side_config.environment_config` and the actual run records through `environment`. Resolve paths relative to the containing manifest. Applicability is specific to the library version, environment, capability, and contract version.

## Shared resources and provenance reuse

Validate model weights, tokenizer, configuration, and source data during preparation. On reuse, confirm access and the actual loading/reading association. Reference existing preparation records across sides and cases. Recheck affected contents when a resource changes, provenance conflicts, or integrity becomes suspect.

| Resource | Preparation | Reuse check |
|---|---|---|
| Read-only shared directory or immutable snapshot with provenance | Reuse its identity and preparation record. | The loaded objects or data reads resolve to that resource. |
| Fixed revision downloaded on each side | Keep the source, resolved revision, completion record, and downloader integrity result. | The actual local load matches its download record. |
| Mutable branch, unestablished cross-host identity, or mixed files | Resolve a fixed version and investigate existing provenance; verify contents where identity remains unresolved. | Load the verified resource and reference the result. |

Record the original repository/data source, resolved revision or snapshot, resource location, and existing verification result. For mirrors, record the actual source URL and returned version. Resolve `main` or `master` during preparation, then reuse the fixed version.

Routine reuse does not require another full scan or hash of shared weights. Frozen inputs similarly reuse selection and tokenization records: check the saved arrays, target mapping, and masks. Source-data provenance and the actual replayed inputs serve different purposes and can both be reused.
