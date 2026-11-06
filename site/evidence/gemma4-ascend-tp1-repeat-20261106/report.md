# Gemma 4 12B: three-repeat NPU text integration pilot

Three fresh sequential TP1 NPU processes completed the same discovery request.
This is a measured single-input repeatability result. The full core evaluation is
incomplete; numerical thresholds and acceptance remain null. The matched H100
Torch 2.9 comparison is pending, so this report contains no GPU/NPU precision result.

The base checkpoint is `google/gemma-4-12B`, revision
`1a2c7729392586f5ecddae8807ee2ef741eabb9a`. The frozen input SHA256 is
`643d29f23d00f97665c5c72d73c63bc20525950b2ca5c50b435140b6066214f0`.
The runtime device name was `Ascend910_9382` on an instance exposing eight NPU
devices; each capture used one device. The observed stack was Torch
`2.9.0+cpu`, TorchNPU `2.9.0.post2`, Transformers `5.10.1`, CANN `9.1.0`
and driver `25.5.2`. The configured/read-back policy used highest FP32 matmul
precision, matmul HF32 disabled, convolution HF32 enabled, deterministic
algorithms disabled, `PYTHONHASHSEED=0` and JIT compilation disabled. The
text-only claim does not cover image or audio execution.

Each process used BF16 parameters, eager attention, seed 0 and one unpadded request
with 128 prompt tokens including BOS plus 32 fixed continuation tokens. One prefill
and 31 cached single-token forwards yielded 32 target distributions. Derived
logprobs used offline FP64 log-softmax of native BF16 logits; this was not FP64
model execution. Original native logits were not retained.

All three distinct repeat pairs had 32/32 paired target scores. In every pair,
logprob MAE, linear P99 and maximum absolute difference were 0.0 nat; both NLL
values were 2.704256948418421 nat/token. The local metadata-only replay reproduced
those target metrics from the recorded values and matched the archived source
report. These observations do not establish general determinism.

The archived source-side summary reports 96 full-vocabulary rows checked across
three captures, with 262144 vocabulary entries per row. Its three pairwise
comparisons each contain 32 positions: top-1 agreement 32/32, mean top-100 overlap 1,
and zero mean/P99/maximum union-token probability RMSE, maximum probability error
and tail-merged JS. JS uses the top-100 union plus one complement category and is
coarsened, not full-vocabulary divergence. Mean top-100/union mass was
0.9294786390914107 (P1 0.5634957276558861; minimum 0.5470973325399145).

**Zero full-vocabulary rows were transferred or numerically replayed locally.**
The distribution values above are archived remote results; the accompanying
standard-library replay only recomputes the target metrics. Archived summary
SHA256 is `95a32147b36d6ae350af58b04c8b733f0279f245916ddc8850e131aa0ef3f5bf`;
the imported evidence archive SHA256 is
`b5070f95c7543a56072a71a5450b01db1112923a7931e7a317f1e561226ed886`.
Hashes identify retained bytes; they do not independently attest execution,
physical allocation or which bytes the loader consumed.

Run `python replay-targets.py selected-targets.json` with Python 3 using the
[replay script](replay-targets.py) and [selected values](selected-targets.json). The replay
checks all three anonymous 32-score pairs, coverage, NLL, MAE, linear P99,
maximum and anonymous row index. Its 1e-12 arithmetic verification tolerance is
not an acceptance threshold. The artifact contains no input text, token IDs,
full distributions, private connection data or device identifiers.

The selected-targets JSON SHA256 is `aca5c3bc9a115cc55a73c9f8b62f45db16e2bb0955717ecf5cb55ca998c799bc`; replay script SHA256 is
`2bda27557d0a267da35f730226c0346300c620996167683fa1bc0a94a654d0ba`. Eight visible instance devices do not establish TP8 or
eight-device coverage. Reserved validation, representative input families and
lengths, cache-off/chunk/batch execution, operator attribution, task quality,
performance, multimodal paths and remaining core work are incomplete.
