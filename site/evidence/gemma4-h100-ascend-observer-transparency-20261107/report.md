# Gemma-4-12B observer transparency on H100 and Ascend

H100 and Ascend each ran four cache-off forwards with one loaded evaluation model on one fixed 160-token input. The common configuration used BF16, eager attention, seed 0 and TP1. This is a scoped two-backend precision comparison.

- Each matched mode agreed at 47 of 47 sampled Top-1 positions in the formal comparison. The hooks-only pair had target logprob MAE 0.033320404863 nat, maximum absolute difference 0.186702472193 nat, and absolute NLL difference 0.00702171989419 nat per token.
- The same pair had mean Top-100 overlap 0.981063829787 and mean complement-merged JS 0.000365748215072 nat. The 98 sampled boundary tensors contained 294,784 paired elements with RMSE 0.0212321369848, maximum absolute error 0.5625 and cosine 0.999800789747.
- The 16 score/distribution pairs include six within each backend and four matched cross-backend modes. The independent CPU check replayed all 752 sampled distribution comparisons and 392 boundary comparisons, and rehashed 388 source files twice. Within-backend raw BF16 payload equality was true for H100 and true for Ascend. Bitwise equality held for 0 of four cross-backend raw payload pairs.
- Both frontend traces reported 863 fired modules, 5,604 invocations and 4,922 scoped signatures in the instrumented mode. These observations capture frontend metadata and sampled boundary values. The scalar BF16 buffer content fingerprint gap remains.

The first Ascend attempt failed before model load at the shared-instance idle check. The successful Ascend capture used a separately bounded manual attempt after the instance became idle. The GPU-only record remains available as an earlier snapshot.

Run `python replay-targets.py selected-targets.json` to replay the anonymous target summaries without native arrays. Full operator, kernel, quality, P01 and original 49-task coverage remain incomplete. The numerical acceptance thresholds are unset.
