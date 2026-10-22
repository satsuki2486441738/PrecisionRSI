# Example: fixed-history inference

## Task

Compare Qwen3.5-9B-Base on NVIDIA H200 and Ascend 910C through their native SGLang stacks. Connections, model snapshots and tokenized validation data are supplied; platform-supported dependency versions may differ.

Use BF16 weights/cache state, TP=1 and one request at a time. Select two records, using 32/64-token prompts and their next three tokens as fixed continuations. Collect continuation scores and distributions, globally and by prefill/first-decode/later-decode stage. Environment preparation, adapter code, jobs and retries are authorized within 30 accelerator-minutes per side, one device each.

## Collection

Use the [inference recipe](../references/inference.md) for native score semantics and the [tool index](../references/comparison.md#tools) for available helpers. Reuse supplied environments and inputs; adapt the native SGLang entrypoint where needed. Capture full-vocabulary-normalized scores before forcing the fixed tokens, preserving each side's cache transitions.

For prompt length `L`, distribution positions `L-1, L, L+1` predict targets at `L, L+1, L+2`. The final prefill row supplies the first continuation score. Pair by these logical positions rather than native output-row names.

## Expected result

Six paired target scores and six distributions; two of each per stage. Report the selected target-score and distribution metrics, actual configurations, coverage, maximum-error coordinates and reproduction paths. Keep full arrays at source and release device allocations when finished.
