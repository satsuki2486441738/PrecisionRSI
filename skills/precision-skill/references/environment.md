# Environment and resource alignment

## Shared implementation

Match upstream releases, including patch versions, of libraries used by both sides for loading, tokenization, computation and scoring. For source installs, match relevant commits/patches. Record platform builds and extensions separately: CUDA, CANN and torch-npu need compatible platform-specific versions, not identical names. Libraries intentionally being compared retain their distinct identities.

Reuse an already approved version pairing. A new mismatch in shared implementation needs an explicit decision on its versions and interpretation before collection/comparison. Numerical agreement alone cannot establish equivalent implementations.

## Associate the environment with the run

Use either a runtime inventory or a locked-installation/immutable-image record linked to the launch and evidence that dependencies remained unchanged. A lockfile alone describes intent; a later inspection alone describes the current environment.

For new runs, record interpreter/environment identity, relevant package releases and runtime builds, imported module locations, platform runtime/driver, and effective numerical settings. Associate every rank with its environment; use a shared inventory when homogeneity is established. Do not import unrelated frameworks or hash entire installations for routine version checks.

For historical runs, examine existing preparation, launcher and change records before proposing recollection. Keep unresolved facts explicit; do not backfill current versions as historical observations.

## Reuse resources

Use the existing project environment manager and preserve its declaration/lockfile. Keep environment, model and data installations at their resource locations.

Resolve model/weights/tokenizer/config and data sources to fixed revisions during preparation. On reuse, verify that actual loads and replayed inputs refer to those resources. Repeat content checks only after changes, conflicting provenance or a concrete integrity concern. Reuse tokenization records; validate saved token IDs, positions, masks and target mapping.

Paths locate resources; repository/revision or verified content establishes identity. Environment metadata and entry-file hashes do not attest to every installed source file or internal arithmetic dtype.
