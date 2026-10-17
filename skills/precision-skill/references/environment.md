# Environment and resource alignment

## Shared implementation

Match shared upstream releases, patches and relevant source commits for loading, tokenization, computation and scoring, including the selected model bridge. Resolve their dependency constraints together. Record compatible platform builds/extensions separately; libraries intentionally being compared retain their distinct identities. Reuse the approved pairing and resolve changes to that pairing before collection.

## Execution evidence

Associate each rank with its actual environment: interpreter, relevant package/build versions, imported module paths, runtime/driver and effective numerical settings. Use a runtime inventory or a locked/immutable installation tied to launch with evidence that dependencies remained unchanged. A lockfile alone is intent; a later inspection cannot backfill historical execution. A shared inventory is sufficient when rank homogeneity is established.

Review existing preparation, launch and change records before recollection. Scope runtime inspection to the components used by the selected execution path and retain unresolved provenance explicitly.

## Resources

Use the project's environment manager and retain declarations/locks. Pin model, tokenizer/config and data revisions; verify actual loads and replayed token IDs, positions and masks. Reuse preparation checks unless resources changed or provenance conflicts. Keep large resources at their existing locations: paths locate them, revisions/content establish identity. Environment records do not prove every internal arithmetic dtype.
