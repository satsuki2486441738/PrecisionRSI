# Environment and resource alignment

## Shared implementation

Declare the comparison scope before selecting versions. For a complete platform-stack comparison, the variables include hardware and its entire dependency implementation. Select an internally compatible, supported stack for each platform and pin its relevant releases, source commits, builds and extensions independently. Shared version numbers or identical framework source are not prerequisites. Conclusions apply to the recorded stack pair; hardware-only or operator-only attribution requires additional evidence.

Align model/data identity, effective model computation, algorithm and objective semantics, precision policy and replay state at the measured boundary. Verify effective reductions, masks, score transforms, clipping and optimizer rules rather than treating matching configuration names as proof. Semantic differences block only affected comparisons; retain observations for compatible stages. Use identical upstream versions only when a narrower shared-implementation comparison is requested.

Reuse the selected profiles and the user's authorized version-selection scope. A supported per-platform version choice within an approved whole-stack comparison does not require renewed approval merely because the versions differ. Use unmodified declared library sources and their supported build/configuration interfaces. Keep historical locally modified stacks labeled as separate source-variant profiles; their results do not establish an unmodified-library baseline.

## Execution evidence

Associate each rank with its actual environment: interpreter, relevant package/build versions, imported module paths, runtime/driver and effective numerical settings. Use a runtime inventory or a locked/immutable installation tied to launch with evidence that dependencies remained unchanged. A lockfile alone is intent; a later inspection cannot backfill historical execution. A shared inventory is sufficient when rank homogeneity is established.

For adapted or wrapped implementations, record the executed method and its resolved operator bindings, following decorators and inherited methods; a selected class name does not establish the kernel path. Compare effective configurations and verify parameter conventions at adapter boundaries when they affect the result, such as a loader’s weight transformation and the receiving module’s interpretation of that weight.

Review existing preparation, launch and change records before recollection. Scope runtime inspection to the components used by the selected execution path and retain unresolved provenance explicitly.

## Resources

Use the project's environment manager and retain declarations/locks. Pin model, tokenizer/config and data revisions; verify actual loads and replayed token IDs, positions and masks. Reuse preparation checks unless resources changed or provenance conflicts. Keep large resources at their existing locations: paths locate them, revisions/content establish identity. Environment records do not prove every internal arithmetic dtype.
