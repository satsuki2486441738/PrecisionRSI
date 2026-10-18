# Collect observations

## Align

Match model/checkpoint/tokenizer, input IDs/positions/masks, algorithm and reductions, numerical policy, and replay state. For training, include optimizer groups, moments/step, accumulation and clipping. Match actual random inputs when enabled.

Give each execution setup a distinct side ID and select the pairs to compare. Record hardware, runtime and framework per side. Pin compatible dependencies independently per setup and retain native operators. Align upstream versions when a shared-implementation comparison is requested. Reuse existing environments, resources and valid captures.

## Run

Resolve connections, native entrypoint, allocation, output path and timeout from the task configuration. Check device identity and capacity, then run the requested stage within the available cumulative budget.

Save coordinates, arrays, effective settings, imported package/runtime versions and per-rank completion together. Validate at the source; compare there or transfer large arrays through the agreed artifact store with size/hash verification. Return small reports locally.

Release resources and settle cumulative usage before retrying. Fix collection/launch errors within the task's budget. Use the [tools](tools.md) when captures are ready.

## Adapt an unfamiliar implementation

Use the installed framework version's documentation and source to locate batch preparation, native execution and the requested capture boundaries. Execute through its native entrypoint, operators and state transitions.

Map outputs to the selected task's logical coordinates: request/token positions and vocabulary IDs for inference; sample/action or parameter coordinates for training. Resolve packing, sharding, padding and replicas. Retain native storage dtypes and distinguish captured values from derived quantities.

Keep collector adapters with the task and adapt observations to the selected [tool interface](tools.md). Report unavailable requested observations with their affected coverage. For task-specific objectives, use the [metric definitions](metrics.md). Extract reusable framework guidance when repeated use warrants it.
