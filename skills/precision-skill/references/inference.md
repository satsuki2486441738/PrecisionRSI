# Inference comparison

## Model and serialization

Anchor model weights, tokenizer, and configuration to their official repository and fixed revision. Confirm actual loaded identities using preparation and loading records; reuse shared resources according to [environment](environment.md).

For the standard single-turn scoring case, select the first user message and its following assistant response, retaining a system message when present. Use the official serialization. If a Base model has no official chat template, agree on a common serialization instead of applying an Instruct template implicitly.

Choose complete samples within both sides' agreed length limit before execution. Preserve them without truncation; record original, included, and excluded counts and reasons. Keep the existing data source and split. Independent train/validation/test splits matter to generalization claims, not to a fixed-input numerical comparison.

Keep BOS, role markers, separators, and terminators required by serialization. Score assistant body tokens only, excluding special tokens, user/system content, and padding. Establish body-token boundaries and validate the target positions and mask after tokenization. Save the evidence for `target_scope`; a body span alone does not prove special-token exclusion. The bundled core rechecks saved arrays and does not load a tokenizer or invent a special-ID system.

## Forward and scoring

Use the same weights, token history, positions, attention/score masks, parameter storage dtype, quantization, forward autocast policy, and scoring definition. Check effective execution values. CUDA/CANN kernels and native fusion may differ; retain the chosen native paths rather than changing precision or selecting eager execution simply to reduce the error.

Run in eval mode with autograd disabled. Teacher forcing consumes the complete fixed sequence and scores its actual next-token targets. Start with full-forward, align prediction position t−1 with target t, and use raw-model scores. Record precision conversions in the scoring path separately from forward autocast.

Optional greedy generation can locate trajectory divergence. Once generation branches, equal sequence indices no longer imply a common token history. Report native-stack differences; attributing them solely to hardware or to an operator requires additional localization evidence.

Use the distribution protocol below for new next-token consistency cases. The existing schema-v2 target-score protocol remains available for historical artifacts and auxiliary NLL; it does not implement this distribution protocol. Use [comparison](comparison.md) for evidence and report states.

## Next-token distribution protocol: top100-union-v1

This protocol fixes k=100 per side at each agreed prediction position. GPU is the numerical reference. It compares raw-model distributions before temperature, top-k/top-p sampling filters, or other logits processors. Preserve the full-forward, eval/no_grad, cache-off and FP32 full-vocabulary log-softmax policy above. The diagnostic selection of top-100 must not change model execution or sampling settings.

Freeze the position set and its selection before collection. For the current assistant-body case, retain the existing nonspecial assistant-body prediction positions rather than silently expanding to prompt positions. A target token label is needed for auxiliary NLL, not for distribution comparison. This observation scope does not claim coverage of every sequence position.

### Required observations and pairing

- Bind model/weights/tokenizer/config, shared environment, vocabulary size and token-ID mapping, full input histories, position IDs, attention masks, selected prediction positions, and completion/rank evidence. Identify coordinates by sample identity and prediction position; verify the prefix seen at that position, not just its ordinal index.
- At each position save each side's ordered 100 token IDs and their full-vocabulary-normalized logprobs. Sort by descending score and break exact ties by ascending token ID. Fail this fixed-k profile if the vocabulary has fewer than 100 entries. Preserve argmax token ID and the actual selected probabilities without top-100 renormalization.
- Form S as the union of the two top-100 sets (100 to 200 unique IDs). Save **both sides' logprobs for every ID in S**, not only their own top-100. Align by ID, never by rank within the lists. Missing counterpart scores are missing coverage, not zero probability.
- Each side saves its own top-100 probability mass and its probability/log-mass outside S, plus the normalization and scoring implementation. Compute tail mass stably from the full distribution, preferably with log-sum-exp over the complement; preserve enough precision and rounding metadata to validate unit mass. Do not hide a negative tail with unconditional clamping or add epsilon to the KL denominator.
- Save auxiliary ground-truth token IDs/logprobs when NLL is requested, even when those tokens lie outside S. Keep these outside the set used for top-100 metrics.

Two independent sparse top-100 files do not contain the required cross-scores. Before collection, choose either retained full score arrays that can be queried offline, or an explicitly authorized replay after exchanging token IDs. Replay must use the same state/history and bind the results to the run; do not combine unverified repeated observations. Retained full arrays can be temporary while the final handoff is sparse. Exchange alone cannot recover discarded scores.

### Metrics and aggregation

Let p be GPU probabilities and q NPU probabilities at one position. Define p_other=1-sum(p_i for i in S), and q_other analogously, computed with the stable policy above. Natural logarithms are used throughout.

| Metric | Per-position definition | Across-position report |
|---|---|---|
| Coarsened KL, GPU to NPU | sum over S plus other of p_i * (log(p_i)-log(q_i)) | Mean, global linear P99, maximum and its coordinate. |
| Top-1 agreement | Equality of the argmax token IDs. | Fraction of agreeing positions and numerator/denominator. |
| Top-100 overlap | Size of the two top-100 sets' intersection divided by 100. | Mean. |
| Probability RMSE | sqrt(sum over S of (q_i-p_i)^2 / size(S)); excludes other. | Mean, global linear P99, maximum. |
| Maximum probability absolute error | max over S of abs(q_i-p_i); excludes other. | Mean, global linear P99, maximum and position/token ID. |
| Coverage | Each side's own top-100 mass, and each side's union mass. | Each separately: mean, global linear P1, minimum. |
| Auxiliary target scoring | NLL on the actual labeled targets and signed delta NLL; optional PPL=exp(NLL). | Token-weighted, with target count. Keep distinct from distribution metrics. |

Each unique prediction position has equal weight. Compute per-position RMSE before aggregation so positions with larger unions are not overweighted. Merge all samples/batches/ranks before percentiles; do not average rank percentiles. Linear percentile at fraction a uses h=a*(N-1) and interpolation between floor(h) and ceil(h). At N=1 it is the sole observation. Reject empty coverage, duplicates, nonfinite observations, and missing/extra coordinates rather than intersecting sets.

For the Qwen full-forward case, preserve the original FP32 logprobs and record their full-vocabulary probability mass in FP64. Require its deviation from one to be at most 5e-6, then normalize over the **full vocabulary** in FP64 for distribution metrics and tail aggregation. This corrects finite-precision normalization and is never a top-100 renormalization. The tolerance checks score serialization/normalization, not GPU/NPU precision acceptance; auxiliary NLL continues to use the original target logprobs. Bind this numerical policy in the case contract before execution.

For KL, a zero reference mass contributes zero; positive reference mass with zero candidate mass gives infinite divergence and cannot yield a finite passing diagnostic. Preserve finite logprobs to avoid mistaking probability exponentiation underflow for a true zero. Record invalid normalization or numerical failures explicitly. There is no acceptance epsilon or threshold in this profile; any serialization tolerance must be separately justified and fixed before comparison.

Name the metric `top100_union_tail_kl`, not full-vocabulary KL: merging all unobserved tokens into one category hides within-tail redistribution and gives a lower bound on full-distribution KL. Top-100 agreement does not establish complete-distribution agreement. Top-1 mismatch examples should include both candidates' probabilities and the GPU first/second probability margin; a small margin is context, not an automatic exemption.

Thresholds remain null (`no_thresholds`). This document specifies the protocol; use a case-specific collector and distribution validator, since the bundled comparator does not implement it. Do not put these observations into schema v2 or relabel historical target-token reports as top-100 results.

### Related work and scope of adoption

- [llama.cpp perplexity implementation](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/perplexity.cpp) reports KL mean/quantiles/max, same-top-token rate, PPL ratios and target-token probability RMS. Its KL implementation uses compressed reference logprobs and a low-probability cutoff. Our union-plus-tail KL and union probability RMSE are adaptations, not exact reproductions of that implementation.
- [AWQ, evaluation](https://arxiv.org/html/2306.00978v5) and [SmoothQuant, evaluation](https://arxiv.org/html/2211.10438v6) use perplexity and/or downstream task scores to assess model quality. These complement numerical agreement and do not replace it.
- [msprobe inference comparison](https://github.com/Ascend/msprobe/blob/master/docs/zh/user_guide/accuracy_compare/infer_compare_result.md) documents KL and tensor error metrics. Its generic thresholds are not imported into this top-100 protocol.

Sources reviewed on 2026-10-15. Freeze source revisions when implementing a collector based on upstream code; links to development branches describe the reviewed guidance, not immutable execution dependencies.

## Cache coverage

Treat cached decoding as a separate capability. Fix the common token history, positions, initial state, and a trustworthy layout mapping.

- Evolving each side's cache along the same token history measures accumulated state differences.
- Injecting the same cache for one step isolates a same-state decoding comparison.

Record native versus injected scope. Unobserved cache behavior remains outside coverage even if scoring is available. The bundled full-forward comparator does not implement cache validation.

## Localization references

The following official guidance provides diagnostic methods, not universal thresholds:

- [msprobe inference guide](https://github.com/Ascend/msprobe/blob/master/docs/zh/best_practices/infer_debug_guide.md): configuration/input alignment, reproducibility, and bad-case localization in section 4.
- [msprobe training guide](https://github.com/Ascend/msprobe/blob/master/docs/zh/best_practices/train_debug_guide.md): numerical versus convergence results, checklists, and first-step versus long-run localization.
