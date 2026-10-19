# Worked example: fixed-history inference across two stacks

## Request

> Compare Qwen3.5-9B-Base inference on an NVIDIA H200 and an Ascend 910C, using each platform's native SGLang stack. The connections, model snapshots and tokenized validation data are available in the task workspace. Compare the hardware and its software stack together; platform-supported dependency versions may differ.
>
> Use BF16 weights and cache state, TP=1 and one request at a time. Select two validation records with at least 67 tokens. Use prompt lengths 32 and 64, followed by the next three tokens from each record as fixed continuations. Collect continuation target scores and full output distributions through native cached execution. Report global results and the prefill/first-decode/later-decode breakdown.
>
> You may prepare environments, write adapters, submit jobs and retry within a cumulative accelerator-time budget of 30 minutes per side, using one device per side. Return numerical metrics, actual configurations, coverage and artifact paths.

## Expected execution

Read the [inference recipe](../references/inference.md) and select tools from the [tool index](../references/comparison.md#tools). The agent carries out the following work.

| Stage | Agent action | Result |
|---|---|---|
| Establish conditions | Inspect the supplied resources. Assign side IDs `h200` and `ascend910c`; verify a common checkpoint/tokenizer/config identity. Freeze sample IDs, prompt/continuation token IDs, position IDs and masks once. Record BF16 state settings, TP=1 and the requested coordinates as common conditions; record native dependencies/backends per side. | Shared comparison conditions, frozen inputs and per-side configurations |
| Prepare execution | Reuse compatible environments or select supported platform combinations. Call `runtime_probe.probe_runtime` for ordered imports and required interfaces in the selected interpreter. Check native worker initialization and perform a small execution in the allocated device environment. | Runtime identity, working launch configuration and a checked capture boundary |
| Integrate collection | Inspect the installed SGLang entrypoint and implement the task adapter. Map native request rows to sample IDs and reconstruct complete vocabulary rows. Use `FixedHistory` to save FP32 log-softmax before forcing each prescribed token; let the native engine advance its cache. | Collector plus per-side logprob arrays, token histories, stage evidence and completion records |
| Validate and export | Map native scheduling events to logical prefill/decode events and check them with `validate_history`. Verify normalization, vocabulary identity, actual settings and coordinate coverage. Use `export_records` to create distribution positions and target scores; add manifest settings and rank evidence. | Validated manifests with six distributions and six target scores per side |
| Compare | Return the small target-score manifests and rank evidence for `compare_target_scores.py`. Keep full distribution arrays at source. On each host, use `query_distributions.py inventory`, exchange inventories, then run `query` to obtain union scores and tail statistics; run `compare` on the small query results. Apply the same metrics to the requested stage groups. | Global and stage-level reports using the recipe's metric definitions |
| Deliver | Return configurations, metrics, coverage and artifact paths. Release allocations and report cumulative resource usage. | Completed numerical comparison |

The agent chooses interpreter paths, launch commands, adapter implementation and artifact locations after inspecting the environment. It asks when an unresolved choice changes this task's comparison conditions or exceeds its resource budget. Collection and launch failures are corrected within the supplied budget.

### Coordinate check

For a prompt of length `L` and continuation `[u0, u1, u2]`, the requested rows are:

| Capture | History used to predict the target | Distribution prediction position | Target position |
|---|---|---|---|
| Last prefill row | Prompt | `L-1` | `L` (`u0`) |
| First decode row | Prompt + `u0` | `L` | `L+1` (`u1`) |
| Later decode row | Prompt + `u0, u1` | `L+1` | `L+2` (`u2`) |

Thus the 32-token request uses distribution positions `31, 32, 33`; the 64-token request uses `63, 64, 65`. Each stage group contains two paired distributions and two paired target scores.

## Expected delivery

The final response identifies both execution configurations and links to:

- Target-score results: each side's NLL, absolute NLL difference, and logprob MAE/P99/maximum with its coordinate.
- Distribution results: tail-merged Jensen–Shannon divergence, top-1 agreement, probability RMSE, maximum probability error and mass coverage, globally and by stage.
- Coverage: six requested/collected/paired targets and six distributions, with any unavailable observations accounted for explicitly.
- Frozen input identities, runtime/configuration evidence, manifests, comparison reports and resource accounting.

Measured values come from the generated reports. Once the requested metrics and coverage are available, this example's task is complete. Internal layerwise sampling is a subsequent task entered after user confirmation.
