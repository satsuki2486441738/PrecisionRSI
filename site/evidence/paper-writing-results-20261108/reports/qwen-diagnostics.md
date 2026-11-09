# Qwen3.5 and Qwen3.8 follow-up: existing results

Evidence snapshot: 2026-11-07T09:11:27.106592+00:00. **Interim delivery; full follow-up final delivery remains incomplete.**

The user reported that the NPU machine is temporarily unavailable and requested
results already obtained. This package freezes completed, independently reviewed
evidence without connecting to a native host, executing a model or modifying the
original arrays. Q35 NPU cache has **not launched**. Its absence is preserved as
missing numerical coverage. DeepSeek's earlier applicable delivery stays closed
with its explicit numerical gaps and is included unchanged.

## Current findings

| Study | Completed evidence | Result and limit |
| --- | --- | --- |
| Q38 packed-loader diagnosis | Selected original TP8 repair and validation | Fixed-history first-ID agreement 2/32 → 31/32; long tails 16/16; structured tails 23/24. Residual differences remain. |
| Q38 controlled heldout | 9 examples, 36 independent complete-candidate forwards per peer | GPU 6/9 correct; NPU 6/9; predictions agree 5/9. Maximum shared full-vocabulary JS 0.2456284425591637 nat. |
| Q35 backward diagnosis | Four boundary-factor cells, all 80 gradients per cell | Common hidden and cotangent jointly change one target's sign; nonzero residual and interaction remain. No unique operator or hardware cause. |
| Q35 historical SFTTrainer | Continuous/restored and two independent reset processes per policy | Native level1: 448/448 continuation and 704/704 per reset comparison exact. Default: 298/448 continuation exact; reset differences remain. |
| Q35 controlled heldout | 9 examples, 36 independent complete-candidate forwards per peer | Both 9/9 correct; predictions agree 9/9; raw firstmax agrees 92/92 rows. Shared maximum JS 0.003943495456112716 nat. |
| Q35 GPU cache | 4 selected 8K/32K examples, 36 stateful calls | Complete and independently raw-replayed. Selected 8K full/split residuals remain; selected 32K observations are exact on GPU. |
| Q35 NPU cache | No native execution | Missing; paired cache metrics cannot be computed. |
| DeepSeek | Previously delivered applicable study | 191/275 paired tensor cells and all 3,168 target coordinates. Training/framework/operator gaps remain. |

These controlled probes do not establish broad benchmark accuracy, numerical
acceptance, equivalent backend arithmetic, general determinism or a hardware
fault. The same labels across models do not create an unbiased model ranking.

## Frozen configurations and comparison contract

Qwen3.5 revision is `68c46c4b3498877f3ef123c856ecfde50c39f404`.
The actual original inference model has **9,409,813,744 parameters**; the earlier
training-plus-LoRA count 9,410,485,488 is a different inventory. The completed P1
heldout peers and GPU cache all inventory their model parameters as BF16. This
does not assert that every intermediate activation or operator uses BF16.

| Q35 P1 setting | GPU | NPU |
| --- | --- | --- |
| Torch | 2.9.0+cu126 | 2.9.0+cpu with torch_npu 2.9.0.post2 |
| Transformers | 5.20.2 | 5.20.2 |
| Native layout | singleton logical device | singleton logical0 within original allocated eight-chip scope |
| Attention | SDPA | SDPA |
| Torch deterministic algorithms | false | false |
| Native NPU deterministic level | not applicable | 0 |
| GPU TF32 matmul | false | not applicable |
| Seed | 2317 | 2317 |

Historical Q35 training uses separate policies and versions. The backward study
uses Transformers5.20.2, while historical SFTTrainer uses Transformers5.3.0,
TRL0.29.0, PEFT0.20.0 and Accelerate1.12.0. NPU nativelevel1 results are not relabeled
as original-default P1 results.

Qwen3.8 revision is `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
Its GPU automatic layout and NPU TP8 with the case-local packed-loader repair
remain different configurations. A matched GPU TP8 control is not available.
Q35 singleton results do not remove this Q38 layout confound.

All heldout scores sum every causal answer-token log-probability after FP64
normalization over the full 248,320-token vocabulary. They use no top-k
renormalization or answer-length normalization. Four candidate forwards per
example are independent full forwards; correlated answer rows are not independent
examples. Cache calls below are stateful continuations, not independent full
forwards. No after-divergence autoregressive row is treated as a same-history pair.

## Qwen3.8: supported repair, retained differences

The version-scoped defect is in packed gate/up weight loading. At retained
coordinates, 144/144 loaded values follow the ordinary contiguous loader and
126/144 disagree with the intended packed sharding. The case-owned candidate
corrects 48 packed loads per rank while preserving 610 ordinary loads. Selected
byte checks cover 73 weights on each of eight ranks, not every checkpoint value.

For the selected original fixed-history workload, first-ID agreement changes
from 2/32 to 31/32. One unique-maximum disagreement remains at prediction position
128: candidate1558 versus GPU220. Long-context tails agree16/16 and structured
tails23/24. Selected32K full-forward JS still reaches0.12437185718941084nat.

Nine untouched controlled examples yield6/9 correct on each side but only5/9
matching predictions. In English/Chinese8K, GPU selects1 and NPU0; in the32K
counterparts GPU selects0 and NPU1. Candidate0 is expected in this frozen set.
Both are correct on code8K and wrong on code32K.

| Frozen example | GPU choice | NPU choice | GPU expected-answer margin (nat) | NPU expected-answer margin (nat) | Max shared full-vocabulary JS (nat) |
| --- | ---: | ---: | ---: | ---: | ---: |
| val-en-note | 0 | 0 | 1.812500000 | 1.281757105 | 0.00854674462 |
| val-zh-note | 0 | 0 | 1.562500000 | 1.808948888 | 0.00520905561 |
| val-code-note | 0 | 0 | 5.392223555 | 5.430896619 | 0.0131899204 |
| val-english-8192 | 1 | 0 | -0.353478690 | 0.274105890 | 0.0156305605 |
| val-english-32768 | 0 | 1 | 0.206085428 | -1.229025993 | 0.0545625514 |
| val-chinese-8192 | 1 | 0 | -1.041251496 | 0.587019370 | 0.150506544 |
| val-chinese-32768 | 0 | 1 | 0.177787201 | -1.315313242 | 0.0127844931 |
| val-code-8192 | 0 | 0 | 2.474293262 | 4.712185630 | 0.0841425311 |
| val-code-32768 | 1 | 1 | -0.066078944 | -0.847568390 | 0.245628443 |

The shared maximum JS is0.2456284425591637nat; independent raw computation is
0.2456284425591238nat. Both values remain in results.json; neither silently
replaces the other. All36 paired arrays differ. Equal correct counts do not
establish equal per-example outcomes or equal distributions.

Original selected32K full and native half-cache calls took850.7691448s and
849.3834689s. Separate uninstrumented128/8192 trials used two warmups and five
measured all-eight-rank maximum spans, with medians4.230705299s and208.703346185s.
There is no matched stockTP8 timing baseline, so causal repair overhead is not
measured. Instrumented heldout collection duration is not stock inference latency;
allocator windows and host snapshots are not whole-host peak memory.

## Qwen3.5: backward boundary and historical continuation

The retained target is layer7 v_proj LoRA-B[302,1]. GPU original-default is
-0.00012838603288400918 in every cell. The NPU nativelevel1 cells are:

| Hidden / cotangent | NPU target gradient |
| --- | ---: |
| own / own | +0.00025825362536124885 |
| own / common | +0.00016923870134633034 |
| common / own | +0.0000825625320430845 |
| common / common | -0.00017406661936547607 |

Their interaction is-0.00016761422739364207; the common/common residual versus
GPU is-0.00004568058648146689. Both common factors together change this target's
sign. The replay uses the original interval8..15 incoming cotangents and original
lower graph; upper gradients remain original. This is a bounded boundary-factor
result, not a full-model counterfactual or unique operator attribution.

Each cell compares80 gradient arrays/671,744 elements, with40 arrays still unequal.
Own/own to common/common maximum difference falls from0.0010940991342067719 to
0.00034349970519542694, RMSE from7.539258111976912e-06 to3.6053882702599115e-06 and
cosine rises from0.999241361934603 to0.9998153794548595. Residuals remain. Historical
default controls have exact retained states/gradients/logits but a two-float32-ULP
loss span; the strict gate failed and factorial was not executed under that
policy. The separately admitted nativelevel1 strict gate passes. No new
cause-supported training repair candidate has been selected, so no new
multitrajectory repair-efficacy result is claimed.

Historical SFTTrainer captures use four fresh processes per NPU policy. All128
checkpoint-entry arrays are exact. Continuation and reset comparisons are:

| Comparison | Default exact arrays | Default maxabs | Nativelevel1 exact arrays | Nativelevel1 maxabs |
| --- | ---: | ---: | ---: | ---: |
| continuous-restored | 298/448 | 0.00163333357 | 448/448 | 0 |
| continuous-reset1 | 390/704 | 0.00369060121 | 704/704 | 0 |
| continuous-reset2 | 379/704 | 0.00490692258 | 704/704 | 0 |
| reset1-reset2 | 379/704 | 0.00375337107 | 704/704 | 0 |

Each default-to-level1 same-step2 phase has192/448 exact and256 unequal arrays.
Thus exact within-policy level1 continuation does not establish cross-policy or
GPU/NPU equivalence. Source sets and guards nativelevel at phase entry, but no
independently saved per-phase policy-transition/Torchflag receipt exists.
Historical scalar target scores lack independent raw full-vocabulary replay.
These limitations remain explicit even though the bounded state-array study is
independently complete.

Earlier tie-aware inference retains45/48 equal first IDs, overlapping maximizer
sets48/48 and exact tied sets34/48. Overlapping maxima do not imply equal
probabilities, and shape-dependent changes occur on both peers.

## Qwen3.5: completed P1 heldout

The original-default singleton peers completed36 independent candidate forwards
across9examples each, including the frozen8K/32K task examples. There are36 raw
full-vocabulary arrays,92 causal rows and22,845,440 logits per side. Both select
9/9 expected complete answers; all9 predictions and92 raw firstmax/tied sets agree.
Maximum raw logit difference is0.4375; shared maximum JS is0.003943495456112716nat,
independent maximum0.0039434954561127665nat. The independent review passes928checks
with no findings. These nine fixed examples are not a general accuracy benchmark.

| Frozen example | GPU choice | NPU choice | GPU expected-answer margin (nat) | NPU expected-answer margin (nat) | Max shared full-vocabulary JS (nat) |
| --- | ---: | ---: | ---: | ---: | ---: |
| val-en-note | 0 | 0 | 3.125000000 | 3.125000000 | 1.8848741e-05 |
| val-zh-note | 0 | 0 | 0.062500000 | 0.062500000 | 3.77635321e-05 |
| val-code-note | 0 | 0 | 2.240006069 | 2.239378333 | 2.0685233e-05 |
| val-english-8192 | 0 | 0 | 4.145888676 | 4.110327878 | 0.000400622802 |
| val-english-32768 | 0 | 0 | 4.329881235 | 4.230713229 | 0.000403614185 |
| val-chinese-8192 | 0 | 0 | 1.231208518 | 1.769979774 | 0.00394349546 |
| val-chinese-32768 | 0 | 0 | 2.016158426 | 2.104644617 | 0.000631714166 |
| val-code-8192 | 0 | 0 | 4.951624731 | 4.949208034 | 0.000464345864 |
| val-code-32768 | 0 | 0 | 4.619204510 | 4.509934234 | 0.000611281544 |

The NPU heldout recorder duration is1279.516s and driver duration1260.893s.
Both include capture-specific work; neither is an uninstrumented throughput
benchmark. These actual complete forward results remain valid while the machine
is unavailable; they do not fill the separate missing NPU cache experiment.

## Qwen3.5: GPU-only cache/state results

Four selected8K/32K examples from a12-example bank produced36stateful native calls
and573,440processed prompttokens. There are768state hash observations;96selected
raw arrays contain52,199,424bytes, while672observations retain hashes only.
Hash-only inventory is not numerical raw-array coverage. Same-shape repeats and
full/chunk observation transparency are exact at the retained scope.

| Example | Fullcache vs chunk logit maxabs | Selected state maxabs | Same-shape repeats |
| --- | ---: | ---: | --- |
| structured-domain0-8192 | 0.125 | 0.140625 | exact |
| structured-domain0-32768 | 0.0 | 0.0 | exact |
| structured-domain1-8192 | 0.390625 | 0.375 | exact |
| structured-domain1-32768 | 0.0 | 0.0 | exact |

The independent GPU cache review passes3704checks without findings. Its whole
export has285files/164,128,206bytes. Native recorder time is256.287796s and driver
time232.256413s; these are instrumented capture durations. Exact selected32K
observations on GPU do not establish whole-cache equality, universal cache
correctness or a GPU/NPU cache comparison.

**NPU cache remains not launched.** The last actual read-only probe at
2026-11-07T08:46:46Z failed before SSH authentication/key exchange with255; remote
Python did not execute. The user then reported temporary machine unavailability.
Current remote process/device state and partial source deployment remain unknown.
The existing admission is unconsumed. No new probe, deployment or model submission
is performed for this package.

## DeepSeek existing delivery

The unchanged [DeepSeek final report](deepseek41.md) and its
[aggregate evidence](../sources/deepseek41-precision-20261106.json) are
included. Its pinned revision is1bae0e1088993b9b82de08f59de5be5a34e42170, with
191/275paired tensor cells and all3,168target coordinates. Target MAE is0.280315nat;
all six controlled candidate first choices agree. Missing reference-cache,
masked-padding, training/framework, determinism and component scopes remain.
Original GPU mixed-format arithmetic and adapted NPU BF16/TP8/EP8/offload are
separate configurations. Applicable delivery is complete; all-agreed numerical
chains and numerical acceptance remain false.

## Figures

The five supplemental figures below reuse the exact independently reviewed bytes.
They display aggregate evidence and do not perform new raw/model replay.

Complete-answer outcomes and distribution residuals [retained private evidence or external source; not bundled]

Selected packed-loader boundary growth and margins [retained private evidence or external source; not bundled]

Q35 boundary-factor and historical continuation [retained private evidence or external source; not bundled]

Historical gradient and actual stored-update evidence [retained private evidence or external source; not bundled]

Q38 observed configuration cost and limitations [retained private evidence or external source; not bundled]

## Reproducibility and remaining work

[Machine-readable results](../sources/qwen-diagnostics.json) contain per-example complete-answer
scores/margins, shared and independent JS values, policy-separated training
metrics and selected GPU cache summaries. Source identities [retained private evidence or external source; not bundled]
retain repository-relative authority paths, SHA256 and byte counts. Original raw
arrays, checkpoint/model resources and native logs remain in their existing
private runs and are not duplicated into this small package.

Run `python -I -B verify.py` from the package root to check every included file
against MANIFEST.json and validate aggregate counts, margins and scope boundaries.
This is package integrity and aggregate consistency, not independent raw-array
replay. Optional plotting commands, requiring NumPy and Matplotlib, are:

```sh
python -I -B replay/four-themes/plot.py --data replay/four-themes/measurements.json --output NEW_FOUR_THEME_DIRECTORY
python -I -B replay/paired-heldout/plot-paired.py --data replay/paired-heldout/paired-measurements.json --output NEW_PAIRED_DIRECTORY
```

The final Qwen follow-up still needs original NPU cache execution/verified export,
paired root and independent cache raw replay, integrated final delivery review and
fresh exact final NPU/service cleanup after availability returns. Existing
historical cleanup evidence does not prove current host state. Conditional P2
serving is not invoked without the plan's required dependency. No publication,
new model computation, altered historical evidence or full-task completion is
claimed by this interim snapshot.


Export note: commands mentioning private packages describe the retained private workflow; this aggregate bundle includes only the verifier documented in its root README.
