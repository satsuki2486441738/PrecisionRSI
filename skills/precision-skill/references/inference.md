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

Use [comparison](comparison.md) for schema-v2 collection, NLL/MAE/P99, and acceptance.

## Cache coverage

Treat cached decoding as a separate capability. Fix the common token history, positions, initial state, and a trustworthy layout mapping.

- Evolving each side's cache along the same token history measures accumulated state differences.
- Injecting the same cache for one step isolates a same-state decoding comparison.

Record native versus injected scope. Unobserved cache behavior remains outside coverage even if scoring is available. The bundled full-forward comparator does not implement cache validation.

## Localization references

The following official guidance provides diagnostic methods, not universal thresholds:

- [msprobe inference guide](https://github.com/Ascend/msprobe/blob/master/docs/zh/best_practices/infer_debug_guide.md): configuration/input alignment, reproducibility, and bad-case localization in section 4.
- [msprobe training guide](https://github.com/Ascend/msprobe/blob/master/docs/zh/best_practices/train_debug_guide.md): numerical versus convergence results, checklists, and first-step versus long-run localization.
