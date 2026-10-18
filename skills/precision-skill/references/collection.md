# Prepare collection

## Align

Match model/checkpoint/tokenizer, input IDs/positions/masks, algorithm and reductions, numerical policy, and replay state. For training, include optimizer groups, moments/step, accumulation and clipping. Match actual random inputs when enabled.

Pin compatible dependencies independently per platform and retain native operators. Align upstream versions when a shared-implementation comparison is requested. Reuse existing environments, resources and valid captures.

## Run

Resolve connections, native entrypoint, allocation, output path and timeout from the task configuration. Check device identity and capacity, then run the requested stage within the available cumulative budget.

Save coordinates, arrays, effective settings, imported package/runtime versions and per-rank completion together. Validate at the source; compare there or transfer large arrays through the agreed artifact store with size/hash verification. Return small reports locally.

Release resources and settle cumulative usage before retrying. Fix collection/launch errors within the task's budget. Use the [tools](tools.md) when captures are ready.
