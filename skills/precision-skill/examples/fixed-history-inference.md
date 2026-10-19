# Example: fixed-history inference

## Task

Compare Qwen3.5-9B-Base on NVIDIA H200 and Ascend 910C through their native SGLang stacks. Connections, model snapshots and tokenized validation data are supplied; platform-supported dependency versions may differ.

Use BF16 weights/cache state, TP=1 and one request at a time. Select two records, using 32/64-token prompts and their next three tokens as fixed continuations. Collect continuation scores and distributions, globally and by prefill/first-decode/later-decode stage. Environment preparation, adapter code, jobs and retries are authorized within 30 accelerator-minutes per side, one device each.

## Expected process

Follow the [inference recipe](../references/inference.md) and [tool index](../references/comparison.md#tools).

1. **Prepare:** verify common model/tokenizer/config identity; freeze token IDs, positions and masks. Record common conditions and each side's actual configuration. Select compatible environments, probe imports/interfaces and check a small execution in native workers.
2. **Collect:** implement the SGLang adapter and use `FixedHistory` to capture FP32 logprobs before forcing each continuation token. Preserve native cache transitions and map request rows to logical coordinates.
3. **Validate and compare:** check histories, normalization, vocabulary and coverage; export existing manifests. Compare target scores and distribution queries, keeping full arrays at source. For prompt length `L`, distribution positions `L-1, L, L+1` predict targets at `L, L+1, L+2`.
4. **Deliver:** return measured metrics, actual settings, coverage, maximum-error coordinates and artifact paths; release allocations and report resource usage.

## Expected result

Six paired target scores and six distributions; two of each per stage. Report the recipe's target-score and distribution metrics globally and by stage, accounting for unavailable observations. Include frozen inputs, runtime evidence and comparison reports.
