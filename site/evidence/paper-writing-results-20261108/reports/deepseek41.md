# DeepSeek-V4.1-Flash-1017 GPU/NPU precision evaluation

Evidence snapshot: 2026-11-06T16:56:46.080812Z. Applicable delivery with explicit numerical gaps.

The retained complete-model captures support a descriptive comparison of 191
of 275 expected tensor observations and all 3,168 target log-probabilities.
They do not establish numerical acceptance or equivalent backend arithmetic.
The GPU uses the original mixed-format reference implementation; the NPU uses
an explicitly adapted BF16 execution configuration with a different parallel
layout and offload policy. The mean absolute target log-probability difference
is 0.280315 nat. All six controlled candidate questions have the same first
choice, while some token distributions and lower candidate ordering differ.

The real-backbone output-head study and the complete uninstrumented NPU
performance matrix are independently reviewed. The final native task returned
0 and exact owned-process cleanup is verified. Training and several framework
chains have actual admission evidence but retain numerical gaps. This delivery
closes the measured and applicable diagnostic scope; it does not declare every
agreed numerical chain complete or label a numerical acceptance pass.

## Frozen model and execution configurations

The only model release in this evaluation is
`deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. The original 84-file resource
inventory totals 510,311,045,870 bytes. Original FP8, FP4 and UE8M0 storage is
preserved. Existing un-suffixed DeepSeek-V4.1-Flash resources, old TP8 conversions
and Ascend W4A8 are separate releases/configurations and do not substitute for
this revision.

| Configuration | GPU reference core | NPU adapted core |
| --- | --- | --- |
| Execution | Official standalone original inference sources | vLLM 0.30.0+empty / source-pinned vLLM-Ascend 0.30.0rc2 text path |
| Parallel layout | TP4 original reference | TP8 / EP8; one complete model on the secondary allocation |
| Arithmetic | Original FP8/FP4 mixed-format native kernels | Explicit original-storage expansion and BF16 native arithmetic |
| Engram | Original checkpoint shard mapping | Two original-storage mmap aliases per rank |
| Offload | Reference-owned placement | cpu_offload_gb=48; UVA and pinned weight offload disabled |
| Scheduling | Original explicit reference cache positions | Synchronous, one sequence, prefix caching disabled |
| Long request | Full 32,768 tokens; native multi-token continuation gaps retained | Full 32,768 and exact [16,384,16,384] reference partitions |

The fresh NPU load verifies all 46,080 owned converted matrices using immutable
coordinate proofs and checks both original mmap aliases per rank. Exact
conversion coordinates establish the declared represented weights; they do
not establish equivalence to original quantized activation/GEMM arithmetic.
The primary allocation's offload42/nonmmap backward and repeatability results
remain separately labeled. Logical card separation on the secondary allocation
does not create independent host RAM: its 16 logical devices share a 1 TiB
host cgroup, so complete BF16-offloaded models are serialized.

## Input, state and coordinate contract

Inputs are frozen once against the pinned tokenizer, with discovery/reserved
partitions and 128/512/2,048/8,192/32,768-token bins. The independent inventory
contains 275 tensor observations, 3,168 target coordinates and 18 controlled
candidates across six questions. Stress and procedural sequences are bounded
numerical probes, not a general natural-language accuracy benchmark.

Each cache comparison checks actual consumed IDs, absolute positions and
native partition receipts. The NPU core has 216 successful native requests:
five repeatability controls, 168 inference requests, 24 structured requests,
one fixed teacher trajectory and 18 candidate requests. Across eight ranks,
1,728 terminals and 3,760 actual forwards match frozen histories and source
identities. Eighteen requests consume 32,768 tokens: six inference full,
six inference exact split, three structured full and three structured split.
Short inference, loading, kernel fixtures and scheduling are not counted as
those long-context calls.

The fixed trajectory consumes all 32 frozen continuation tokens using 33
samples: prefill forces teacher[0], then 32 cached forwards consume teacher[0:32].
The retained row at 128+s predicts the following token after consuming
teacher[s]. Both peers therefore share the same complete consumed history.
These measurements do not describe unconstrained free generation. No
post-divergence free-generation row is treated as a common-history pair.

Target and candidate probabilities use shared full-vocabulary logsumexp in
FP64. Distribution diagnostics preserve the union's true probability mass and
merged tail; top-k renormalization is not used. Candidate likelihoods sum all
candidate token terms, including multi-token choices. Forty actual post-FFN
mHC boundaries retain positions × four residual streams × 5,120 coordinates;
stream axes are not collapsed.

## Strict coverage

| Stage | Expected tensor cells | GPU retained | NPU retained | Paired |
| --- | ---: | ---: | ---: | ---: |
| blocks | 40 | 40 | 40 | 40 |
| fixed-decode | 32 | 32 | 32 | 32 |
| inference | 168 | 102 | 168 | 102 |
| padding | 6 | 0 | 0 | 0 |
| repeatability | 5 | 5 | 5 | 5 |
| structured | 24 | 12 | 24 | 12 |

Overall coverage is 191/275 paired cells. GPU gaps are 78 actual multi-token
reference-cache rejections plus six masked-padding gaps. NPU gaps are the same
six masked-padding cells. Shared omissions remain missing. The standalone/public
request interfaces do not provide the admitted masked-padding semantics; an
unmasked synthetic substitute is not counted. NPU full/split observations do
not fill missing GPU reference cells. The full frozen target and candidate
score inventories are paired separately from tensor coverage.

## Numerical observations

| Target group | Coordinates | MAE (nat) | P99 (nat) | Maximum (nat) |
| --- | ---: | ---: | ---: | ---: |
| All | 3,168 | 0.280315 | 3.593114 | 10.432424 |
| long-targets | 288 | 0.278096 | 3.284949 | 5.938382 |
| quality | 2,304 | 0.269107 | 3.637231 | 10.432424 |
| structured | 576 | 0.326255 | 3.608924 | 7.153308 |

Global GPU NLL is 1.9395359408692914 nat/token; NPU NLL is
1.9448933733189466. Their absolute difference is 0.005357432449655253 nat/token.
The maximum target difference occurs at quality/reserved-18-512,
position 0, token 16. These are descriptive score differences without an
acceptance threshold, and the mean NLL difference does not summarize the tails.

At selected positions, first-ID agreement is 264/264 full-forward, 144/144
single-token cache, 29/32 fixed teacher trajectory and 36/48 structured full.
Eleven of the twelve structured first-ID disagreements have overlapping exact
top-logit tied sets; one has disjoint tied sets. The three fixed-trajectory
mismatches have disjoint tied sets. Exact tied sets themselves agree at 33/48
structured positions. First-ID agreement is consequently distinct from tied-set
agreement and from task accuracy.

All six candidate first choices agree. Probe 3 reverses the lower-ranked
candidates, so complete rankings are not identical. Forty mHC boundaries show
configuration-dependent error accumulation: layer-0 RMSE is 0.001659955 and
layer-39 RMSE is 1.763738305; corresponding cosine values are approximately
0.999723 and 0.982670. A high final cosine does not erase earlier boundary
errors or maximum-coordinate outliers, and this does not isolate a hardware
cause.

Each side's full-versus-cache contrasts also show shape/state/arithmetic
sensitivity. GPU token-cache selected top-1 agrees at 144/144 positions while
maximum logit difference reaches 5.766824335. NPU token cache also agrees at
144/144 with maximum 2.96875. NPU exact split agrees at 264/264 inference
positions, maximum logit difference 3.5625; structured split agrees at 40/48,
maximum 18.0. GPU multi-token-cache cells remain absent. These contrasts do not
by themselves diagnose a cache implementation defect.

## Output-head transparency and causal boundary

The terminal core retains selected-position and native sampling-head logits;
its largest selected/native receipt is 0.0625. A synthetic original-checkpoint
TP8 head fixture first isolates shape sensitivity: M1/M4 last rows match native
M1, while M49 differs by 0.00390625 and exactly reproduces selected M49. That
synthetic fixture alone is not a real-backbone cause result.

Six subsequent complete-model controls A/B0/B1/C0/C1/D use the identical frozen
32,768-token long-00 history, actual seed 0 and logprobs=8. A/D execute the
native path; B passively retains original tensors; C runs the unchanged
selected-head collector alongside passive retention. All six native sampled
tokens and recorded top-eight log-probabilities agree. B0/B1 and C0/C1 repeat
exactly; B/C normalized hidden states and native M1 head outputs agree exactly.
The frozen collector changes no retained backbone hidden values in these calls.

Same-hidden native head replay at M1 and M4 matches the native last row exactly.
M49 reproduces the entire selected-M49 output exactly, with exact repeats.
At the last row, 28 of 129,280 logits differ from native M1, maximum 0.03125.
Both outputs retain the same singleton top-logit set, token 223. Shared FP64
full-vocabulary normalization gives total variation 1.5009041237698725e-12,
maximum probability error 1.4475574490121938e-12 at token 19,860, and probability
RMSE 4.963466793291496e-15. These small probability effects are a property of
this sample's distribution; they are not a general guarantee for logit errors.

Shared top100-union plus merged-tail metrics have a separate scope: their
maximum probability error is 7.88258e-13 at token 223. Their reported JS=0 is
finite-precision cancellation and does not imply exact full-distribution
equality. Production sparse log-probabilities and independent FP64 normalization
also use different arithmetic; the maximum retained difference is 4.72558e-7.

This establishes a head-shape effect for the actual same-hidden long-00,
logprobs=8, seed=0 configuration. It does not explain every original core
observation, whose logprobs=None configuration and maximum receipt 0.0625 are
separate. It does not attribute GPU/NPU differences to hardware. Independent
reviews rehash all 280 canonical transport files (285,486,737 bytes), bind the
976-array native audit and 854 dependent rank comparisons, and verify exact
owned-worker cleanup. Successful controls are reused after the later,
separate timing-stage failure.

## Architecture fixtures

Eighty actual native router calls cover 40 layers × two samples × four
positions. Independent FP32/FP64 sqrtsoftplus references applied to actual
retained native logits validate all 320 top-six sets, including cutoff ties.
The GPU expert sets agree at all 320 positions. Expert-ID aligned weight
maximum absolute difference is 5.066394805908203e-7. Native transformed router
scores were not retained, so the independent score arithmetic remains an
explicit policy. FP32 gate projection on common original BF16 inputs/weights
is distinct from original GPU activation arithmetic. The release's default
num_hash_layers=0 supplies dynamic routing; no hash-routing coverage is added.

Four original Engram capsules cover 384 distinct checkpoint rows and 98,304
scalar coordinates. Independent raw E4M3/E8M0 decoding followed by one BF16
rounding matches the actual original GPU lookup and all eight NPU gathered
replicas exactly. Native TP8 hash-head ranges are contiguous. Replicated
allgather outputs are not independent examples; this does not cover complete
tables, native hash generation, full backbone or 32K execution by the capsule.

Bounded fixed-index MLA admission preserves original causal window and selected
index geometry. Metadata eventually succeeds after explicitly recorded ABI
adjustments, but the actual kernel rejects ori_sparse_indices as A5-only.
There is no numerical output from that invocation; other layer capsules are
unattempted. Production regular/packed branches and the fixture's metadata
ratio0/kernel1 sentinel differ explicitly. This closes the exact invocation's
compatibility record, without a production-wide or hardware-wide unsupported
claim. Indices are not removed merely to obtain a successful call.

The integrated NPU 32K backbone executes native Indexer/compressor paths,
including compressed-source construction and native sparse-index selection.
This source-bound integrated execution does not supply separate same-input
NPU Indexer scores/candidate/cache values, compressed-key construction or query/
output-projection numerical capsules. GPU sampled Indexer score/selection replay
is separate and does not independently reconstruct compressed keys. Those
per-component paired scopes remain unmeasured. The exact MLA ABI rejection
cannot establish that these other components are unsupported.

Four bounded native mHC preop calls cover attention/FFN at layer 0 on two
samples, with retained pre/post/comb values and local references. The NPU
collapsed output y is finite, but the corresponding GPU y was not retained;
there is no paired-y numerical comparison. These bounded operator scopes are
separate from the 40 paired full-core actual post-FFN four-stream boundaries.

## Training, framework and native-chain applicability

GPU and NPU both perform actual complete-model forward, native train-mode
requests and loss.backward admission. Both original inference execution paths
produce outputs without grad_fn and reject backward. On NPU all eight ranks
attempt 1,032 floating-parameter flags: 798 enable and 234 reject inference-tensor
grad flags. The actual 128-token target-455 backward is rejected. Native BF16
cross-entropy 11.5625 and independently recomputed full-vocabulary FP64
11.56298152865497 are separate arithmetic policies.

No numerical LoRA updates, optimizer trajectory, checkpoint continuation or RL
training is established. Trainer.call_model_init, SFTTrainer constructor,
GRPOTrainer constructor and verl.HFModelConfig actually reject the pinned
model's missing deepseek_v41 Transformers registration on both environments.
These CPU entrypoint outcomes do not count as accelerator training numerical
coverage or prove that a different training implementation is impossible.

The CPU framework registration admissions use separate pinned training
bundles, rather than either final inference-core interpreter: both use
Transformers 5.20.2; GPU uses PyTorch 2.9.0+cu126 and NPU-host CPU admission
uses PyTorch 2.9.0. Trainer/SFTTrainer/GRPO bundles record TRL 0.29.0,
PEFT 0.20.0 and Accelerate 1.12.0; the verl bundle records verl 0.8.0.dev0,
PEFT 0.21.2 and Accelerate 1.12.0. All eight actual entrypoints reject the
missing deepseek_v41 registration before accelerator execution. These versions
scope the registration outcomes only. They do not replace GPU core PyTorch
2.13.0+cu130 or NPU core PyTorch 2.10.0+cpu / Transformers 5.14.1, and no
accelerator training numerical coverage is inferred.

The three native chains are official standalone reference, vLLM and SGLang.
Official GPU execution supplies the retained reference core, but is not an HTTP
server lifecycle. The actual official NPU generate/model/kernel import path
and independent kernel import reject missing tilelang before model construction;
act_quant is not invoked. GPU vLLM actual TP4 construction is OOMKilled 137 at
the 256,000,000,000-byte host limit before successful load/generation. NPU vLLM
supplies the adapted complete-model core. GPU SGLang retains native CPU-memory
OOM evidence even after a separately verified synchronous scheduling change;
NPU SGLang reaches its actual host-table initialization and rejects the
cudaHostRegister/None cudart interface. None of these failures is counted as
serving scores or training numerical success. No HTTP lifecycle claim is made.

## Repeatability and performance

GPU bounded same-process controlled seed resets and a separate fresh-process
replay are exact within their recorded configuration. Primary NPU offload42
same-process controlled-seed 128/2,048 interleaved calls retain exact full-vocab
comparisons; there is no same-configuration fresh-process NPU claim. The eight
native allgather ranks are dependent. Selected core baseline/repeat controls
also agree within each retained side, without proving global determinism.
The separate real-head logprobs=8, seed=0 repeats do not establish determinism
for the original core logprobs=None configuration or a fresh NPU process.

Both performance matrices complete nine planned cells and 63 requests per
side: two warmups and five retained measurements per cell. There is no planned
32K cached-decode timing cell. The final NPU model loads in 389.493 seconds;
its complete recorded lifetime is 5,274.297 seconds. Timing excludes model load.
Independent receipt replay checks all eight ranks: 959 nonempty forwards each
(63 prefills and 896 cached forwards), plus 63 untimed finished-request cleanup
ticks each. All 33 samples and 32 consumed teacher tokens, actual IDs/positions,
seed/prefix resets, synchronization and method restoration are verified.
Extra selected-head hooks, passive taps and block collectors are absent.

| Prompt tokens | Phase | GPU median seconds | NPU median seconds | NPU measured range seconds | NPU phase peak allocated / reserved GiB |
| ---: | --- | ---: | ---: | --- | ---: |
| 128 | prefill | 2.179504 | 4.724649 | 4.649820–5.022988 | 44.657 / 44.787 |
| 128 | fixed 32 cached steps | 8.902484 | 155.729824 | 154.729117–158.820593 | 44.645 / 44.787 |
| 512 | prefill | 2.454958 | 4.956528 | 4.842127–5.001665 | 44.694 / 44.865 |
| 512 | fixed 32 cached steps | 8.988494 | 151.924267 | 150.270602–154.136453 | 44.645 / 44.865 |
| 2048 | prefill | 3.529494 | 5.072352 | 4.977570–5.201864 | 44.843 / 45.197 |
| 2048 | fixed 32 cached steps | 8.910710 | 149.790004 | 148.759870–152.553224 | 44.645 / 45.197 |
| 8192 | prefill | 5.928523 | 5.539316 | 5.450764–6.032738 | 45.832 / 46.330 |
| 8192 | fixed 32 cached steps | 8.992839 | 150.416576 | 149.705736–155.586899 | 44.645 / 46.330 |
| 32768 | prefill | 16.632518 | 9.273984 | 9.025140–9.277603 | 56.591 / 59.297 |

Each cell retains its five individual measurements, mean, sample standard
deviation, range and driver-wall diagnostics in the aggregate archive. NPU
memory columns are the maximum measured phase PyTorch allocator peaks across
ranks/requests, include the loaded baseline and use binary GiB. For cached
decode the phase excludes prefill; separately archived whole-request peaks
include it. These are not host-memory or whole-device peaks. No cold-cache
claim or clean hardware speed ratio is made: arithmetic, layout, offload and
timer boundaries differ between the two configurations.

Actual core runtime receipts identify GPU PyTorch 2.13.0+cu130, CUDA 13.0,
TileLang 0.1.12, Transformers 5.20.2, safetensors 0.8.0 and NumPy 2.3.5. The
standalone original implementation uses its actual interpreter environment;
being installed in a vLLM environment does not make vLLM the execution engine.
Its TP4 model partitions 384 routed experts into 96 per rank on the same
four-rank world, with output all-reduce; it has no independent EP mesh flag.
FP4/FP8 linear operators quantize inputs to FP8, whereas native Indexer queries
and keys have FP4 quantization and other control/compressor paths include BF16
and FP32. No single dtype describes every operation.

NPU actual core receipts identify PyTorch 2.10.0+cpu, vLLM 0.30.0+empty and
Transformers 5.14.1. The +cpu package suffix does not describe device execution:
actual tensors, calls and native receipts run through torch_npu on the NPU.
vLLM-Ascend 0.30.0rc2 identifies the selected hash-pinned source checkout; its
installed distribution build is unknown. The CANN 9.1.0 installation selection
appears in native stderr, but the exact CANN build is unknown. A same-environment
bounded native mHC receipt records torch_npu 2.10.0.post4 and Ascend910_9382; the
core did not separately recapture those package/device strings. Exact Python
versions were not recorded; the NPU source environment exposes a 3.12 ABI.
These unknowns remain unknown rather than being filled from workload defaults.

Actual head input/output and hidden tensors are BF16. Exporting them as FP32
NPY preserves values without changing native arithmetic. Bounded mHC pre/post/
comb tensors are FP32 while collapsed y is BF16; this does not imply every
integrated module executes wholly in BF16 or FP32. Runtime fact review153703Z
binds these statements to 17 original receipt/source files and exact fields.

The NPU worker timer wraps unchanged native execute_model, including offload
and native head, with trailing local NPU / existing TP CPU Gloo synchronization.
Sampler/forcing and post-timer input observation are separate. Generation wall
time includes scheduler work; worker RPC transport is outside that metric.
Cached decode aggregates max_rank(sum of 32 rank-local spans), excluding the
prefill; it does not sum per-step rank maxima. Allocator statistics are
aggregated over the measured phase or whole request as explicitly labeled,
excluding any host/whole-device peak claim. GPU
and NPU arithmetic, layout and timing boundaries differ, so a hardware speed
ratio is not derived. Instrumented core times are not performance measurements.

The initial NPU timing gate failed before any measured trial: a bare default
distributed barrier lazily requested 420,478,976 bytes for an additional HCCL
communicator under the loaded model and raised EL0004/OOM. The exact failure,
CPU reproducer and terminal cleanup are retained. This collector defect does
not invalidate the completed head controls or establish a model failure.

The repaired timer binds the already initialized native TP CPU Gloo coordinator,
synchronizing each local NPU before and after that coordinator's barrier. An
actual no-model TP8 fixture completes 133 synchronizations per rank, guards
against new/default/device groups, rejects identity drift before a collective
and succeeds after exact group restoration. Independent review verifies all
29 exported files, actual barrier source/AST, rank identities and cleanup.
This fixture proves only the synchronization interface. The successor loads
successfully with a fresh 46,080-matrix representation census and 16 mmap aliases,
but its first actual gate rejects a legitimate zero-token scheduler cleanup
step after the native prefill. The retained scheduler dump has empty counts,
total scheduled tokens 0, finished request IDs and speculative_config=None.
The original runner updates request/batch/cache state and returns its native
empty sentinel in this configuration. The wrapper's one-request check was
incorrect; this is separate from the earlier HCCL allocation failure.

A minimal repair passes that original cleanup step through, records it outside
model-forward timing, and preserves forward/sample indices and restoration.
Seventeen CPU regressions and an independent native-source lifecycle audit
support the repair; these are not actual loaded-model gate or performance
coverage. Cleanup may contain native cache device operations, so it is not
claimed to be device-free. The failed successor's sole driver is terminated
only after workers and engine have exited and its exact PID/birth/argv match;
terminal code 143 records this owned cleanup, not a model numerical failure.
Its complete 42-file export (43,081,349 bytes), source admission and released
devices are independently checked. The metadata-only native runner fixture then completes 16 actual empty calls
across eight ranks using the real factory's canonical empty KV metadata. Its
37-file export, exact group and resource cleanup are independently reviewed;
it is not real DeepSeek KV geometry or live-request recovery. The next fresh
loaded-model gate supplies the actual-request evidence described above. The
matrix then completes all 63 requests and exact terminal cleanup, with two
independent audits described above. Completed head controls are reused.

## Integrity, independent replay and cleanup

A 100 MiB per-file export cap first rejected the terminal execution-index file;
a later large-transfer timeout and SSH255 failures are preserved. No model was
rerun for transport. Twenty-five small verified batches recover the canonical
union: 6,013 files, 3,216,418,653 bytes and 353 canonical arrays. Every original
raw rank remains at its native path. A read-only native audit rehashes all
35,905 NPY files totaling 19,919,982,720 bytes and checks finiteness/geometry.
All 31,108 same-coordinate raw peer comparisons match.

Root independent composition binds terminal inventory, source exports and
canonical payloads. It checks 31,792 manifest geometries and 3,760 native-head
geometries, then reconstructs 4,172 canonical-row raw NPY hashes from the local
values. Target/candidate probabilities are independently rederived through
shared tools. Frozen eleven-file comparator closure runs in an isolated
interpreter from an empty directory; a separate fresh replay reproduces the
complete paired result JSON bytes. Retained artifact and authority paths
remain explicit dependencies; this is not claimed as a fully relocatable bundle.

Independent service cleanup rehashes 731 export files and makes new exact UID
queries for seven historical GPU jobs. Their jobs and pods are absent. NPU
SGLang/official/core exact old groups and baseline devices are independently
reviewed. The completed head owner has separate exact PID/birth/group cleanup. The first performance-only owner has exact terminal/export/cleanup evidence.
The final performance lifetime has its own complete 433-file export totaling
432,763,480 bytes. Original-host inventory, every transport file, 28 frozen
sources, 13 production helpers and 20 actual environment keys agree. A fresh
46,080-matrix census has zero unequal coordinates; all 16 original mmap aliases
are validated. Its exact remote group and local continuation chain are absent.
The recorded 16 logical chips have zero AICore activity and no owned running
processes; retained HBM is within 5 MiB of the recorded admission baseline,
not zero device memory. Host under-OOM and OOM-kill counters are zero. These
checks apply to the exact task and observation times, not every later system
state. Separate performance arithmetic and source/cleanup reviews both pass.

## Evidence and delivery scope

Core comparison: compare-ds41-paired-core-frozen-comparison-20261106T120928Z.
Isolated replay: compare-ds41-paired-core-isolated-independent-replay-20261106T121316Z.
Canonical/raw derivation: compare-ds41-core-canonical-raw-derivation-independent-audit-20261106T122435Z.
Descriptive digest: compare-ds41-paired-report-independent-digest-20261106T122726Z.
Three-chain cleanup: compare-ds41-three-chain-service-cleanup-independent-audit-20261106T123733Z.
Real-backbone head: compare-ds41-real-backbone-head-transparency-independent-audit-20261106T134232Z.
Second head review: compare-ds41-real-head-second-independent-cpu-audit-20261106T134714Z.
Timing recovery gate: compare-ds41-performance-existing-tp-sync-independent-gate-20261106T135058Z.
Strict gap ledger: compare-ds41-final-delivery-gap-independent-audit-20261106T140005Z.
Empty-step diagnosis: compare-ds41-performance-empty-tick-native-causal-diagnosis-20261106T141000Z.
Independent cleanup-step source audit: compare-ds41-empty-tick-independent-native-source-audit-20261106T141535Z.
All identifiers resolve under the private case runs directory. Raw traces,
connection details and complete model arrays remain private.

Final performance arithmetic: compare-ds41-performance-complete-independent-audit-20261106T164426Z.
Final source/terminal cleanup: compare-ds41-performance-terminal-source-cleanup-second-independent-audit-20261106T164757Z.
Framework runtime clarification: compare-ds41-final-framework-runtime-clarification-20261106T161637Z.
Framework second review: compare-ds41-framework-runtime-second-independent-audit-20261106T162444Z.

## Reproducibility and remaining numerical gaps

The [allowlisted evidence](../sources/deepseek41-precision-20261106.json)
binds measured scope and exact source-file hashes. The
aggregate archive [retained private evidence or external source; not bundled]
retains the scalar measurements, individual performance trials, runtime facts,
explicit gaps and standalone consistency checker/figure renderer. Its immutable
local Git measurement revision is `d805902e30157e0ecd48325ebe2d9d3d1051a5ad` (also recorded
in the evidence and evaluation history); that revision identifies measurements, not model weights. The original
model revision above remains unchanged. The
[scientific diagnostic figure](../figures/3-deepseek41-precision-diagnostics.svg) shows strict
coverage, target tails, selected-ID agreement and 40-layer mHC error growth.

Public aggregate replay checks summary consistency and reproduces the figure.
It does not reconstruct the private raw arrays or execute a model. The separate
eleven-tool private replay reproduces the full paired comparison JSON exactly,
with explicit retained absolute-artifact dependencies; it is not a fully
relocatable public raw-data package. Source hosts, full traces, datasets and
raw arrays remain private. No push or publication is included.

The history status is partial and acceptance remains unspecified. Missing
GPU multi-token-cache and both-side masked-padding cells, zero numerical LoRA/
optimizer/checkpoint/RL trajectories, CPU-only framework admission failures,
no same-configuration fresh-process NPU determinism, absent separate NPU
Indexer/compressor/projection capsules, bounded MLA/mHC output limitations and
the lack of HTTP lifecycle measurements remain explicit. Thus
`all_agreed_numerical_chains_complete=false` even when applicable delivery and
independent audit close. This is a configuration comparison, not proof of a
hardware defect or a general task-accuracy benchmark.

After final report/archive/history validation and independent delivery admission
are retained, the authorized serial monitor continues automatically to Qwen3.5/
Qwen3.8 supplemental Stage 0. Qwen3.8 retains its original primary eight-chip
scope; completed Qwen captures and paused monitors are reused and preserved.


Export note: commands mentioning private packages describe the retained private workflow; this aggregate bundle includes only the verifier documented in its root README.
