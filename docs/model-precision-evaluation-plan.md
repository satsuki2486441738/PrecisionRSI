# Model precision evaluation and reproduction plan

Use this runbook when a task requests a full-model report or when a confirmed
validation goal requires broader model coverage. The
[NumericBridge workflow](../skills/numericbridge/references/workflow.md) owns
request, execution and delivery. Use the profiles and task inventory below to
plan the expanded measurement scope; a normal comparison or repair starts with
the reduced native case in the workflow.

The intended output is a model-specific precision report with measured scope,
operator evidence, workload impact and execution cost. A model output difference
does not by itself identify an incorrect operator. A completed measurement does
not by itself establish numerical acceptance.

## 1 Start here

1. Read repository `AGENTS.md` and the relevant skill entrypoints. Locate the
   checkout using the working directory and Git root; do not assume the folder
   is named `precision-skill` on every host.
2. Create or locate the private case at `projects/<project>/cases/<case>/`.
   Retain an existing case when continuing its workload. Do not move historical
   runs, model files, environments or datasets.
3. Resolve the workload specification in section 3. Record missing access,
   resources or semantic choices explicitly. Do not substitute another checkpoint,
   model class, dataset, operator or runtime to make a task appear complete.
4. Inventory prior results by task ID, actual settings, coverage and artifact
   availability. Reuse compatible, verified results; rerun only affected scope.
5. Select applicable profiles in section 2, freeze the task matrix, inputs and
   acceptance criteria, then follow the execution order in section 12.
6. Deliver the coverage ledger, comparisons, selected replay evidence and report
   described in sections 13 and 14. Leave an exact next action for unfinished work.

This runbook owns the expanded measurement profiles. Endpoint aliases,
credentials, environment paths, adapters, datasets and dated decisions belong
to private cases.

### Repository components

Paths in this table are relative to the repository root. Load only the component
needed for missing work.

| Component | Responsibility |
| --- | --- |
| [Framework Adapt](../skills/framework-adapt/SKILL.md) | Prepare or adapt native environments and workload execution |
| [Precision Compare](../skills/precision-compare/SKILL.md) | Collect, validate and compare observations using shared tools |
| [Precision Repair](../skills/precision-repair/SKILL.md) | Diagnose confirmed correctness concerns, patch and verify the original workload |
| [Inference recipe](../skills/precision-compare/references/inference.md) | Token alignment, raw distributions and cached execution |
| [SFT recipe](../skills/precision-compare/references/sft.md) | Full/LoRA state, native reductions and training stages |
| [RL recipe](../skills/precision-compare/references/rl.md) | Fixed trajectories, objective dependencies and actor/critic updates |
| [Comparison rules](../skills/precision-compare/references/comparison.md) | Metrics, coordinates, interfaces, execution and evidence tools |
| [Internal capture guide](../skills/precision-compare/references/layerwise.md) | Bounded observation plans, msProbe and collector transparency |

`operator-optimize` is a placeholder. This plan evaluates existing native
implementations; writing a faster kernel is a separate task.

## 2 Define what complete means

For a full-model report, use the **core profile**: identity and inputs, native inference,
applicable operator families, model-level impact, repeatability, resource cost
and replay evidence. Enable additional profiles according to the model's
intended use and the report's claims.

| Profile | Enable when | Required work |
| --- | --- | --- |
| Core | A full-model report is in scope | M, D, I, O, Q, P and E tasks below |
| Training | Training precision is in scope and the checkpoint supports it | S tasks; both full-parameter and LoRA when both modes are claimed |
| RL | Policy optimization is in scope | R tasks with a named native algorithm and backend |
| Distributed | The claimed deployment uses multiple devices | X tasks for every claimed parallel layout |
| Quantized | Quantized weights, activations or caches are claimed | Z tasks for each selected quantization policy |
| MoE | The selected architecture contains routed experts | A-MOE tasks as part of architecture coverage |
| Multimodal | The claim includes image, audio or video inputs | A-MM tasks as part of architecture coverage |

A text-only run through a multimodal checkpoint covers its text path, not the
complete multimodal model. An architecture profile may be explicitly deferred,
but the final report must then narrow its title and claims. Unsupported native
support is a coverage gap, not an architecture-level `not_applicable` decision.

### Native compatibility outcomes

Record unsuccessful native execution as a first-class result for each pinned
model/library/backend/configuration cell. Use the following distinctions:

| Outcome | Required basis | Numerical coverage |
| --- | --- | --- |
| Measured native execution | Actual initialized model and successful claimed entrypoint, with retained observations | Only the observed coordinates and stages |
| Native unsupported | An actual native attempt identifies a missing model registration, backend implementation, operator or explicitly unsupported configuration; diagnose environment and collector causes first | Uncovered for the failed stage |
| Resource boundary | The admitted placement and workload exceed memory or another measured resource limit | Retained successful coordinates only; failed bins stay uncovered |
| Environment or collector failure | Broken dependency, launch profile, transport, instrumentation or adapter contract | Not evidence of native unsupported; repair and retry affected scope |
| Adapted execution | A declared patch, fallback or changed execution policy enables the workload | Separate configuration and observations; original native outcome remains visible |
| Deferred or unattempted | No completed native admission for this configuration | Uncovered; no support conclusion |

For every failure retain the native run identity, model revision, package/source
versions, effective dtype/backend/parallelism/placement, exact invocation and
environment profile, terminal status/return code, failure stage and diagnostic
logs. Pin exported bytes with hashes and verify cleanup of owned workers. Keep
connection details and full logs private; public evidence includes a sanitized
reason, immutable evidence hashes, affected scope and any successful retry link.
An SSH disconnect is a transport outcome: inspect the existing native process
and receipt before retrying, and never submit a duplicate surviving job.

Separate compatibility-audit completion from numerical completion in the report.
List attempted, successful, native-unsupported, resource-limited and unattempted
counts, with their configuration denominators. A completed audit may contain
unsupported cells; those cells do not enter precision acceptance denominators or
contribute zero error. No observation means no numerical score. A narrower
successful input length or adapted path does not establish the failed original
configuration's support.

Do not test the full Cartesian product of all models, libraries, precisions and
parallel modes. First establish a common anchor workload. Add one-variable
contrasts around that anchor and the native production paths being claimed.
Every advertised library must execute its actual workload entrypoint; importing
a trainer or calling its loss helper is component-level coverage only.

## 3 Resolve the workload specification

Create `workload.json`, `matrix.json` and `inputs/` in the private case. These are
**case-owned records**, not an existing `workflow.py` or comparator schema.
The case adapter translates them into the owning tools' documented contracts.
Do not pass this illustrative specification directly to a shared CLI.

```json
{
  "plan_version": "model-precision-v1",
  "case_id": "REQUIRED",
  "profiles": ["core"],
  "model": {
    "repository": "REQUIRED",
    "revision": "REQUIRED_IMMUTABLE_REVISION",
    "model_class": "REQUIRED",
    "modalities_in_scope": ["text"],
    "checkpoint_identity_receipt": "REQUIRED_PATH",
    "tokenizer_identity_receipt": "REQUIRED_PATH",
    "config_identity_receipt": "REQUIRED_PATH"
  },
  "sides": [
    {
      "id": "side-a",
      "environment_record": "REQUIRED_PRIVATE_PATH",
      "execution_profile": "REQUIRED_PRIVATE_PATH",
      "native_entrypoint": "REQUIRED_COMMAND_OR_CALLABLE"
    },
    {
      "id": "side-b",
      "environment_record": "REQUIRED_PRIVATE_PATH",
      "execution_profile": "REQUIRED_PRIVATE_PATH",
      "native_entrypoint": "REQUIRED_COMMAND_OR_CALLABLE"
    }
  ],
  "inputs": {
    "discovery_manifest": "inputs/discovery.json",
    "validation_manifest": "inputs/validation.json",
    "training_manifest": null,
    "rl_manifest": null
  },
  "common_conditions": "REQUIRED_CASE_RECORD",
  "intentional_side_differences": "REQUIRED_CASE_RECORD",
  "resources": {
    "authorized_device_count_per_side": null,
    "job_timeout_seconds": null,
    "total_device_seconds_per_side": null,
    "max_capture_bytes": null,
    "max_transfer_bytes": null,
    "max_disk_bytes": null
  },
  "numerical_acceptance": null,
  "matrix": "matrix.json"
}
```

Replace every `REQUIRED` value before affected execution. An empty resource
authorization is not an unlimited budget. Reuse existing authorization where
it applies; resolve genuinely missing limits before accelerator work. Resolve
the following details in the referenced case records:

| Area | Required decisions and evidence |
| --- | --- |
| Model | Original checkpoint bytes or verified hashes, config, tokenizer/processor, model class, all loaded branches, tied weights and conversion mappings |
| Native stack | Actual hardware and assigned device identity; driver, runtime/toolkit, imported library versions/files, framework entrypoint, backend/kernel selection |
| Numerics | Stored weight/activation/gradient/master/optimizer/cache/state dtypes, autocast, TF32/HF32 or equivalent, reduction policies, quantization and rounding |
| Execution | Eager/compiled/graph mode, attention implementation, fusion, batching, cache/chunking, device parallelism and rank ownership |
| Data | Exact tokens or modality tensors, masks, positions, labels, templates, BOS/EOS/padding and selection coordinates |
| Training | Optimizer, scheduler, loss terms/reductions, gradient accumulation, clipping, scaler, adapters, initial states and data order |
| Comparison | What is held common, which side differences are intentional, expected coordinate inventory and designated reference if one exists |
| Resources | Current allocation, time/device count, retries, storage, transfer, timeout cleanup and stopping rules |

Two devices are peer execution sides unless a separately justified reference is
declared. Higher precision is a diagnostic reference with its own validated
semantics; it is not automatically a ground-truth model.

### Task ledger

Each row in `matrix.json` records `task_id`, profile, concrete configuration,
input split, expected coverage, native command/adapter, dependencies, status,
source run/evidence, replay capability and next action. Separate observation
completion, acceptance and diagnosis:

- `pending`: applicable task has not completed.
- `complete`: declared capture, pairing and verification completed at the stated scope.
- `partial`: some declared observations or modes are missing.
- `blocked`: required resource, native support or evidence is unavailable; state the blocker.
- `not_applicable`: a structural reason makes the task irrelevant; record that reason.
- `deferred`: applicable work deliberately excluded from this release; record the resulting claim limit.

Record `acceptance` separately as `null`, accepted or rejected with the frozen
criterion. Record diagnosis as unresolved difference, validated cause or confirmed
defect with evidence. A `complete` measurement may still have rejected or unknown
acceptance. Track `reused` as provenance, not as a weaker completion status.

## 4 Pin and partition inputs

Use token IDs or modality tensors frozen once by an identified tokenizer or
processor. Do not independently tokenize text on each side and assume equivalence.
Retain source IDs, dataset revision/license where relevant, selection method,
content hashes, masks, logical positions and expected targets.

The following is a **proposed starting design**, not a sufficiency guarantee.
Resolve it into exact input manifests before execution. Existing compatible
inputs can replace these defaults when the replacement scope is explicit.

| Input set | Starting design | Use |
| --- | --- | --- |
| Discovery text | 12 distinct examples across natural language, multilingual text, code/math and structured/repeated content; 4 each near 128, 512 and 2K tokens | Collector integration, diagnosis and policy development |
| Reserved validation text | 48 examples with different source/content from discovery; 12 each near 128, 512, 2K and 8K tokens | Frozen-policy validation and distributions by input family/length |
| Long context | 6 reserved examples near 32K tokens when supported and funded | Chunk/cache/state accumulation; include meaningful late dependencies rather than repetition alone |
| Fixed continuation | 32 predetermined tokens per inference example | Equal-history cached score comparisons; label shorter pilots separately |
| Training | 4 discovery examples of 32–128 tokens; 8 reserved examples of 256–1K tokens; nonempty effective labels and unequal target counts | Loss denominators, accumulation and held-out propagation |
| RL | 4 fixed prompt groups with 4 responses each for a group-relative algorithm, or an explicit algorithm-compatible alternative | Action/mask/objective and update replay |
| Stress | Explicitly enumerated zero/small/large magnitudes, cancellation, layout and boundary cases derived from the real operators | Sensitivity diagnosis; keep separate from representative input rates |

Respect actual context limits and reserved output space. For a shorter supported
context, select and freeze replacement bins and report the missing long-context
scope. Never silently truncate examples into a claimed length bin. Context length
includes the fixed continuation where the model's limit requires it.

Keep validation inputs outside policy selection. A failed validation can motivate
a new candidate, but those inputs then become development data; use a new reserved
set for the next generalization claim. Keep input-level distributions instead of
treating thousands of correlated token positions as independent samples.

Select examples deterministically from pinned source revisions: deduplicate by
normalized content hash, assign disjoint discovery/validation source IDs, and
sort eligible examples by a recorded seeded hash before filling each stratum.
Retain original and truncated token spans; omit examples that cannot meet the
required valid-target/context constraints. Synthetic stress fixtures have exact
generators and seeds and are not counted as representative corpus examples.

## 5 Establish native execution and repeatability

| ID | Procedure | Completion evidence |
| --- | --- | --- |
| M01 | Resolve model/config/tokenizer/processor identities and inspect the loaded module/parameter inventory | Identity receipts; conversion, tied-weight and modality mappings |
| M02 | Initialize actual native runtime, import the model class and selected interfaces, verify assigned device binding, run the smallest model call | Successful terminal receipt and actual imported configuration |
| M03 | Execute the selected native framework entrypoint on the complete in-scope model | Native execution/source identity; output shapes and branch/kernel inventory |
| D01 | Validate identical logical inputs, positions, masks, label shifts and expected coordinates | Input hashes and independently derived inventory |
| D02 | Run one fixed same-state workload 3 times sequentially on each side | Within-side error/variance baseline, seeds and deterministic settings |
| D03 | Compare collector disabled/enabled from the same state and RNG | Output/update differences, fired-boundary inventory and payload size |

For D02 use seed 0 and reset all relevant state between repeats. Later seed
contrasts use 0, 1 and 2 with the same seed within each matched pair. Do not claim
equal RNG streams across hardware unless observed. Native repeatability settings
are recorded variables, not universal promises of bitwise determinism.

Preserve the default native arm when repeatability is nonexact. Test a supported
deterministic execution policy as a separate arm, record the observed setting,
and repeat the original multistep/checkpoint workload plus fixed validation
cases. Measure its cost with collection disabled. Improved repeatability does
not establish improved cross-platform correctness; identify remaining operator
and training differences separately. A restored trajectory that differs by no
more than reset-state repeats is insufficient evidence of a checkpoint defect.

Run these checks in the same execution modes used for capture. A CPU import or a
small-module test does not establish full native execution. Collector installation
must not silently replace an optimized path with an eager fallback. If collection
changes compilation, fusion or numerical behavior, report it and select a less
intrusive boundary or narrower claim.

## 6 Measure inference

The causal decoder procedures below apply directly to decoder language models.
For encoder, encoder-decoder, diffusion, embedding or other architectures, retain
the identity/coverage/attribution principles and define equivalent native stage
boundaries. Mark decoder-only tasks inapplicable with the architectural reason;
do not invent a KV cache for a model without one.

| ID | Procedure | Required observations |
| --- | --- | --- |
| I01 | Cache-off full forward through the complete in-scope model | Raw prediction scores, full-vocabulary-normalized targets/distributions and all valid-target inventory |
| I02 | One native prefill followed by 32 fixed single-token decode steps | Final prefill/first-decode/later-decode rows, absolute positions, cache and recurrent-state metadata |
| I03 | Replay the same complete token sequence using full forward, token decode and chunked prefill/decode | Matched prediction rows across partitions; native state evolution |
| I04 | Compare each request alone with variable-length batches and left/right padding where supported | Logical positions/masks; batch membership and padding side; per-request differences |
| I05 | Repeat I01–I03 across frozen context bins within resource limits | Metrics by input and length; memory, failure/OOM and uncovered-bin records |
| I06 | Contrast native production backend with a controlled higher-precision or alternate-backend path | Exact changed settings; same-input results and original-path identity |
| I07 | In a claimed serving engine, compare cache cold/warm and available chunk/scheduling modes | Native request path, cache-hit evidence, fixed histories and preprocessor scores |

Before a long-context suite, admit the largest claimed input using the exact
native full-forward and cached partition paths, selected output rows and device
placement. An import or short pilot does not establish peak-memory support.
Preserve completed logical-input arrays incrementally in the fresh run, with
input/source/placement identities and an explicit expected inventory. A later
failure must not erase earlier completed observations, and unwritten in-memory
arrays cannot count as retained evidence. A memory-boundary failure is separate
from an unsupported model/backend capability. A placement retry retains the
original result and records the new placement as a distinct configuration.

For I03, first compute the full-forward anchor for the same token sequence.
Then use a single prefill plus token steps, and a prefill split into at least two
unequal chunks plus decode. Select chunk boundaries inside relevant kernel/state
blocks and just before/after them when the implementation exposes those sizes.
Record boundaries explicitly. Unsupported chunking is an engine coverage gap.

For I04 start with two unequal-length requests, compare each alone against the
same logical request in the batch, and exclude padding from paired metrics.
Batch changes may change native kernels. A difference is an observation to
investigate, not automatic proof of an invalid implementation.

### Scores and coordinates

Prediction position `t-1` predicts target token at position `t`. The final prefill
row supplies the first continuation distribution. Retain BOS/templates, active
EOS, truncation and complete request/chunk reorder mappings. Save scores before
sampling constraints/processors while preserving model-defined scaling or
softcapping. Record native score dtype, derivation dtype and offline metric dtype
separately.

Capture all requested target logprobs. For distribution capture, default to all
32 continuation prediction rows plus 16 deterministic valid prompt positions per
example, or every prompt position if shorter. Select prompt positions by fixed
even spacing, including the last valid prompt row, and deduplicate them. This is
sampled prompt-distribution coverage even when target-score coverage is complete.
Full-vocabulary data can stay at source and answer verified union queries.

Do not replace unavailable token scores with zero or renormalize a top-k slice.
If token histories diverge in free generation, compare conditional distributions
only through the common history. Analyze the complete generated outputs separately
under Q tasks.

## 7 Attribute differences to operators

An operator test inventory comes from the actual model path, not a generic list
of expected Transformer names. Include architecture-specific components and the
native fused implementations selected by the measured stack.

| Family | Representative scope |
| --- | --- |
| Input and output | Embeddings, tied/untied LM head, logits scaling and output normalization |
| Normalization | RMSNorm/LayerNorm, epsilon, gamma conventions, dtype order and fused residual/norm |
| Projections and MLP | GEMM/GEMV, QKV/gate/up/down projections, bias, activation/gating and fused combinations |
| Full Attention | QK, scaling, positional transform, causal/padding/sliding masks, softmax, AV and backward |
| Position and cache | RoPE or alternative position encoding, KV writes/reads, state layout and boundary transitions |
| Linear or recurrent paths | Convolution, gates, DeltaNet or other recurrence, state initialization/update, chunk scan and decode kernels |
| Routed experts | Router, selection, dispatch/combine, expert computation and load/layout effects |
| Modality branches | Encoder/frontend, projector, resampler/cross-attention and integration with the shared backbone |
| Training numerics | Objective reductions, gradient reductions/clipping, optimizer updates and mixed-precision storage |

| ID | Procedure | Completion evidence |
| --- | --- | --- |
| O01 | Inventory all executed operator families, variants, dtypes and shapes | Applicable families, actual implementation mapping and explicit gaps |
| O02 | Screen every backbone block boundary on bounded real inputs; sample fixed tokens/channels within payload limits | Per-block error trajectory, sample inventory and transparent collector check |
| O03 | Capture real inputs/parameters and incoming gradients for each selected family/variant | Same-input replay inventory with source hashes and semantic mappings |
| O04 | Replay identical captured operands on both native implementations and a validated reference | Local forward/backward metrics separated from propagated-state differences |
| O05 | Test real-shape/layout and selected numerical/size boundaries | Declared stress grid, finite/nonfinite counts and per-cell results |
| O06 | Isolate a selected model-level finding and intervene on one variable | Reproduced finding, intervention, original-workload revalidation and reserved-input result |

For O02 screen all logical blocks with a bounded deterministic slice rather than
dumping every tensor. For O03 select at least one real shape from each applicable
family and each distinct execution path being claimed, including prefill/decode
variants. Refine early/middle/late layers and the interval with the largest
meaningful increase; report the remaining sampling gaps. A small identical fixture
alone cannot establish real-model operator coverage.

### Same input replay and references

For a disputed boundary, save `input`, parameters/state and, for backward,
incoming cotangent before later mutation. Replay both the side A captured operands
and the side B captured operands on both implementations. This separates local
arithmetic from different upstream states. Restore mutable recurrent/cache state
before each replay and compare state outputs as well as visible activations.

Validate a mathematical/high-precision reference independently where feasible.
Examples are a small FP64 formula with autograd/analytic derivative cross-checks,
or a semantically matched higher-precision native path. For a fused operator,
check masks, epsilon, scale, parameter conventions and backward semantics before
using a decomposed reference. An isolated decomposition is not performance-
equivalent to a fused native operator.

When separating error sources, retain both original values and exactly rounded
native inputs. A high-precision calculation on BF16-rounded operands measures
arithmetic error on those operands; casting a BF16 checkpoint to FP32 cannot
recover unrecorded pre-rounding weights.

For O06 first vary one setting, such as accumulation policy, one operator or one
state boundary. Verify unchanged inputs, parameters and all intended untouched
paths. Compare model-level scores, gradients/updates and cost after intervention.
If root cause stays unresolved, deliver a bounded localization and label it as
such. Completion does not require inventing or finding a defect.

## 8 Measure training

Execute a native training entrypoint with the original model and declared full
or adapter trainable inventory. Keep native labels, reductions, accumulation,
state transitions and optimizers. A loss computed from exported logits is a
diagnostic and does not replace observation of the native training objective.

| ID | Procedure | Completion evidence |
| --- | --- | --- |
| S01 | Align initial model/adapters, optimizer/master state, scheduler and data position | Exact logical values or explicitly justified per-side representation; tied/frozen inventories |
| S02 | Execute one aligned-state update with complete trainable coverage | Effective labels/denominators, native loss, accumulated/synchronized gradients, stored before/after and optimizer state |
| S03 | Use two microbatches with unequal valid-target counts and compare declared accumulation semantics | Raw/backward losses, effective denominators, masks and optimizer-step boundaries |
| S04 | Exercise clipping and mixed-precision scaling if selected | Native norm/coefficient, postclip/unscaled gradients, overflow and applied/skipped update events |
| S05 | Execute 2-step pilot then 10-step independent trajectories from identical initial state | Per-step scores/loss, gradient/update norms and drift; complete snapshots at steps 1, 2 and 10 |
| S06 | Run the next update on both sides from one verified common checkpoint | Same-start state evidence, per-side update and state differences separated from cumulative drift |
| S07 | Compare continuous execution against each side's native save/load continuation | Model/optimizer/scheduler/RNG/data-state checks and resumed-step outputs/updates |
| S08 | Validate chosen policy on reserved training batches and initial adapter seeds 0, 1 and 2 | Frozen policy, paired per-seed results, input-level spread and execution cost |

Keep LR, objective and sample order frozen before observing comparative results.
Choose a training configuration that gives finite nontrivial updates; preserve
divergence as a result if it occurs. The 10-step stage measures short-trajectory
propagation, not convergence or final training quality. Longer training requires
its own hypothesis and scope.

### Capture rules

Collect gradients after accumulation, unscale and required synchronization, before
optimizer mutation; collect postclip gradients immediately before the native
optimizer step when clipping applies. Preserve absent gradients, present zeros,
unchanged parameters and applied/skipped optimizer events. Store native dtypes
separately from exported dtypes. Actual update is stored `after - before`, not
the theoretical update formula.

For LoRA retain rank, alpha/scaling, target modules, bias/modules-to-save, dropout,
merge/quantization state, exact initial A/B values and frozen-base unchanged checks.
With zero-B initialization, zero A gradients/updates are valid coverage and must
not be dropped. Cover all adapter coordinates, not only nonzero tensors.

For full-parameter S02 and S06, complete trainable gradient/update coverage is the
target. If resource limits force sampling, label it partial full-parameter
coverage, inventory expected coordinates and report the limitation. Use streaming
comparison or source-side queries before deciding to transfer entire checkpoints.

S07 has two distinct meanings: same-platform native continuation and cross-platform
common-state replay in S06. One does not establish the other. Equal restored target
scores do not establish equal subsequent gradients or updates. Preserve differences
after equal entry state, including within-platform repeatability observations.

## 9 Measure RL when enabled

RL frameworks, training backends and rollout engines are separate axes. Begin
with fixed trajectories; online rollout quality is a later measurement with
different controls. Choose GRPO/PPO or another named algorithm and adapt tasks
to its actual objective rather than treating algorithms as interchangeable.

| ID | Procedure | Completion evidence |
| --- | --- | --- |
| R01 | Freeze trajectory/group/action identities and old/reference policy versions | Prompts/responses, action masks, EOS/truncation, groups and policy/state version mappings |
| R02 | Recompute native current action scores and objective from shared dependencies | Old/reference scores, rewards, advantages/returns/values, normalization and effective denominators |
| R03 | Cover active objective branches on native device workloads | Clipped/unclipped decisions, KL placement, value clipping if present, zero-variance group handling and masks |
| R04 | Run one aligned actor/critic update and a short fixed-trajectory continuation | Training-stage coverage, optimizer/state identity, ordering and stored updates |
| R05 | When online RL claims are made, run controlled rollout/reward/training with repeated seeds | Native loop completion, versions/staleness, reward/task outcomes and total cost |

Mark injected versus recomputed dependencies. Shared injected advantages isolate
the actor update; they do not prove equivalence of reward generation or native
advantage computation. A trajectory with no active clipping does not cover the
active branch merely because a CPU unit test covers its formula.

For asynchronous systems preserve rollout-policy and update-policy versions,
off-policy correction/rejection weights, filtering/packing and redistribution.
Pair the same logical actions after these transforms. Do not use relative loss
error as the main diagnostic when the native objective is near zero or changes sign.

## 10 Cover architecture and deployment extensions

| ID | Applicability and procedure | Evidence |
| --- | --- | --- |
| X01 | Distributed: compare a declared single-device or logical baseline with each claimed TP/PP/DP/FSDP/ZeRO/CP/EP layout | Shard/replica ownership, logical reconstruction, native reductions and complete rank receipts |
| X02 | Distributed: execute accumulation, save/load and native update at selected layouts | State mapping, synchronization, optimizer ownership and continuation |
| Z01 | Quantized: pin calibration data/method, converted bytes, scales/zeros/groups and quantized tensor coverage | Conversion identity and same-input dequantized/reference contrast |
| Z02 | Quantized: compare native quantized operator and complete model prefill/decode/training where claimed | Error and quality/cost deltas; weight/activation/KV/state quantization separated |
| A-MOE01 | MoE: capture router scores, top-k expert IDs/weights, ties/margins, load and dropped/rerouted tokens | Routing agreement and expert/dispatch/combine mappings |
| A-MOE02 | MoE: compare natural routing and diagnostic fixed-routing replay | Routing versus expert-arithmetic contributions; native unfixed model verification |
| A-MM01 | Multimodal: freeze raw assets and exact processor outputs for every claimed modality | Processor/version identity, shapes, resolution/rate and token/tensor mappings |
| A-MM02 | Multimodal: compare encoder/frontend, projector and fused backbone outputs on paired tasks | Modality-boundary coverage and task outputs; image/audio/video length/resolution strata |

Do not silently modify model configuration, such as tied weights, expert count,
image resolution or attention type, to fit a backend. A derived model is a
separate configuration and must be named as such. Mark unexecuted modalities or
parallel layouts deferred/blocked and narrow the report claims accordingly.

## 11 Measure task impact and cost

| ID | Procedure | Completion evidence |
| --- | --- | --- |
| Q01 | Evaluate frozen validation examples with teacher-forced target scores and distribution metrics | Input-level and length/family summaries; mismatch examples and top-1 margins |
| Q02 | Execute native free generation or the architecture's native output task on the same reserved inputs | Generated tokens/outputs, exact task metric implementation and qualitative failures |
| Q03 | For a claimed policy improvement, compare native baseline and frozen candidate on the same reserved workload | Paired quality/precision deltas and uncertainty; no policy selection on these inputs |
| P01 | Time uninstrumented baseline and candidate on matched hardware allocations and workloads | Prefill/decode or train-step latency, throughput, peak memory, warmup and synchronization policy |
| P02 | Record total evaluation cost separately | Preparation, compilation, captures, retries, transfer, diagnosis and cleanup |

For language models use a versioned teacher-forced corpus NLL plus a small
declared task set with executable scoring, such as exact-answer extraction or
code tests. These are separate outcomes. Free-generation text agreement is not
a substitute for numerical pairing. Do not claim broad task-quality preservation
from a few hand-picked prompts.

Starting performance design: at least 2 warmup runs and 5 measured runs after
warmup for a fixed workload, unless the execution budget requires a clearly
labelled smaller pilot. Record device synchronization and measured interval;
separate compilation/model loading from steady state. Use uninstrumented runs
for headline speed, with a separate collector overhead measurement.

Report medians and the observed spread; aggregate task-quality and numerical
effects at input/request level. For sufficiently many independent inputs, use a
paired input-level bootstrap or a declared equivalent uncertainty estimate.
With too few independent cases, report the observations and avoid significance
claims unsupported by sample size. Report precision improvements together with
latency/memory costs on the original workload.

## 12 Sequence the expanded evaluation

1. **Resolve and reuse:** freeze profiles/specification; audit existing evidence,
   runtime/resource identities and expected inventories. Complete M/D admission.
2. **Short native pilot:** execute I01–I04 and a bounded collector trial. Correct
   integration defects before expanding payload or context length.
3. **Representative coverage:** extend to reserved inputs, length bins and claimed
   native engines. Capture bounded block trajectories and operator inventories.
4. **Attribution:** run identical-input/cotangent replays and validate references.
   Isolate at least one selected material finding, or substantiate observed
   agreement within the tested scope.
5. **Training and extensions:** run selected S/R/X/Z/architecture tasks in stages,
   carrying accumulated resource usage forward. Stop on the declared limits.
6. **Freeze and validate:** freeze any selected precision policy, execute reserved
   validation and paired performance work, retaining failed/slower outcomes.
7. **Verify and deliver:** complete E tasks and publish only selected public
   summaries. Keep raw model/data/trace artifacts private.

Run independent sides concurrently only where their resource allocations permit;
dependencies, mutations and shared-state transitions remain ordered. Recheck
idle/owned devices immediately before launch. Use scheduler-assigned device
identities and verify native runtime binding; do not assume visible ordinal 0 is
the allocated device.

Every native job has a timeout, terminal status, complete logs and a cleanup
receipt. Charge failed attempts and retries to the same authorized budget.
Submission is not completion. After timeouts, inspect owned process/job state
before retrying. Never terminate unrelated users' jobs or services.

Do not automatically upgrade a native framework when an import fails. Determine
whether environment initialization, a task-owned overlay or an adapter is missing.
Changed native implementations create a new comparison configuration. A numerical
repair returns to the original workload with unchanged input identity.

## 13 Metrics and verification

Use the owning shared tool implementations and their current schemas. The
definitions below make reports interpretable; they are not permission to replace
shared arithmetic with a case-local implementation.

| Observation | Required measures and interpretation |
| --- | --- |
| Targets/actions | Each side's NLL where meaningful; logprob MAE, P99, maximum and logical coordinate; global valid-target count |
| Distributions | Top-1 count/fraction; top-100 union plus exact complement category JS divergence; probability RMSE/max error; each side's own/union probability mass and secondary top-100 overlap |
| Tensors | Signal norms, L2 error, RMSE, cosine, maximum error and coordinate; expected/observed/paired elements; dtype/structure findings |
| Training | Native objective terms/denominators, all requested gradient/update/state stages, present/absent/zero inventories and applied/skipped events |
| Nonfinite | NaN/+Inf/-Inf counts and coordinates, unmatched observations and paired-finite subset metrics; do not discard nonfinite values silently |
| Repeatability | Within-side and cross-side results shown separately, using matched initial conditions |
| Cost | Measured native durations, throughput and memory; setup/retry/transfer costs separately |

Tensor metrics use stable FP64 reductions. If `e = ||a-b||2` over `N` paired
elements, RMSE is `e/sqrt(N)`. If relative L2 is added, report the denominator:
`e/||a||2` and `e/||b||2` are different quantities. A zero denominator produces
`null` with a reason. Cosine is undefined if either norm is zero. Near-zero
elementwise relative errors need an explicit scale/floor and cannot replace
absolute error and signal norms.

NLL is the global valid-target mean of negative full-vocabulary logprob. Native
loss keeps its actual reduction and auxiliary terms. P99 uses all paired errors
with linear interpolation at `0.99*(N-1)`. Aggregate tensor metrics concatenate
logical coordinates; do not average tensor cosines or rank percentiles.

Distribution metrics operate on original normalization. A top-100 union plus one
complement category is coarsened, not full-vocabulary divergence. Legacy KL
results retain their original direction, coarsening and normalization; do not
relabel them as JS. Full-vocabulary metrics require the original complete rows
or sufficient exact query statistics.

### Verification tasks

| ID | Procedure | Completion evidence |
| --- | --- | --- |
| E01 | Validate both sides against an independent expected inventory | Identity, dtype, semantic coordinate, mask, rank and coverage checks |
| E02 | Independently check selected maximum-error coordinates and representative reductions | Original retained values/hashes and reproduced metric checks |
| E03 | Replay selected validators/comparisons from a portable evidence bundle | Source/export hashes, replay specification and matching expected report fields |
| E04 | Audit final claims and evidence retention | Complete ledger, unresolved gaps, acceptance status and readable artifact locations |

Two sides can omit the same tensor and still pair perfectly; derive expectations
from the task/model inventory. An offline replay verifies exported comparisons,
not fresh accelerator execution. A report without retained arbitrary-coordinate
arrays can still preserve historical measurements, but must state the narrower
replay capability and any recollection needed for new analyses.

### Shared command patterns

Run from the repository root with a Python interpreter containing the maintenance
dependencies. `SIDE`, paths and collector arguments below are placeholders to be
replaced by the resolved case. Read actual schema help before constructing files.
Use an isolated maintenance environment for `python -m pip install -r
requirements-dev.txt`; this does not install native GPU/NPU workloads or their
framework dependencies. Those remain bound to their declared environments.

```sh
python -B skills/precision-compare/scripts/tool_help.py --list
python -B skills/precision-compare/scripts/tool_help.py --full artifact_io
python -B skills/precision-compare/scripts/tool_help.py --full record_run
python -B skills/precision-compare/scripts/tool_help.py --full workflow

python -B skills/precision-compare/scripts/record_run.py --execution EXECUTION.json --output-root RUNS --side SIDE --task capture --timeout SECONDS -- COLLECTOR ARGS
python -B skills/precision-compare/scripts/validate_artifacts.py --contract CONTRACT.json --manifest MANIFEST.json --output NEW_VALIDATION.json
python -B skills/precision-compare/scripts/compare_target_scores.py --contract CONTRACT.json --side-a A.json --side-b B.json --output NEW_TARGET_REPORT.json
python -B skills/precision-compare/scripts/compare_distributions.py --contract CONTRACT.json --side-a A.json --side-b B.json --output NEW_DISTRIBUTION_REPORT.json
python -B skills/precision-compare/scripts/compare_tensors.py --contract CONTRACT.json --side-a A.json --side-b B.json --output NEW_TENSOR_REPORT.json
python -B skills/precision-compare/scripts/compare_tensors.py --mode nonfinite --contract CONTRACT.json --side-a A.json --side-b B.json --output NEW_NONFINITE_REPORT.json
```

Use `query_distributions.py` for source-side distribution queries and
`tensor_artifacts.py` for logical shard mappings. Use `workflow.py` for a resolved
dependency plan. Its `status` operation inspects reuse; `run` can execute missing
native work and must stay within authorization. Case-owned adapters supply
scheduling, native capture and semantic mappings. These tasks are not all
implemented by one generic command.

Exit 0 from a comparator means metrics were computed; it does not mean a numerical
tolerance passed. Until a criterion is agreed and frozen, use `thresholds: null`
and report differences. Software verification tolerances are not model acceptance
tolerances.

## 14 Deliverables and release gate

The private case should expose the following records. Names are case conventions,
not new mandatory shared-tool schemas.

```text
case/
  workload.json              resolved model, peers, profiles and resources
  matrix.json                task/configuration ledger and expected coverage
  inputs/                    frozen manifests and selection metadata
  adapters/                  parameterized native capture and execution adapters
  runs/
    <side>-<task>-<UTC YYYYmmddTHHMMSSZ>/
  REPORT.md                  report and evidence index
  HANDOFF.md                 exact continuation state
```

Keep reusable environment declarations under `projects/<project>/environments/`.
Model/data resources remain at their verified locations. Raw arrays may remain
in source runs or an immutable artifact store; keep retained hashes, byte counts
and source-to-export mappings. Never overwrite historical run bytes or reuse a
past run directory for new validation.

The report contains:

1. The exact model/revision, modalities and workloads covered, actual execution
   peers and intended comparison variables.
2. A task/configuration coverage table with expected/observed counts, sample
   counts, length/shape bins and explicit unsupported/deferred cells.
3. Numerical results by inference stage, operator family, layer/state boundary
   and training step, together with within-side repeatability.
4. Selected findings with observation, local replay/reference, causal evidence,
   original-workload impact and reserved-input validation clearly distinguished.
5. Precision, task-quality and performance effects of each evaluated policy,
   including failed, slower and non-improving outcomes.
6. Numerical acceptance criteria/status, limitations, immutable evidence and
   replay scope. No threshold means no acceptance claim.

**Release gate:** all required applicable cells for the selected profiles are
complete, or the release explicitly lists remaining cells and narrows its claims
to a scoped report. Identity, logical pairing and evidence validation must pass.
Unsupported native execution, missing critical observations or changed semantics
cannot be hidden behind aggregate metrics. A report may publish measured
differences with unknown acceptance or unresolved diagnosis.

`HANDOFF.md` gives the next agent: repository/case paths; frozen workload/matrix
identities; current statuses; exact commands and execution profiles; valid reusable
receipts; owned active jobs/processes; resource usage and remaining authorization;
artifact availability; blockers; and the first safe next action. Include private
connection references, never secrets. A next agent should not need this chat.

For public history, follow [the site instructions](../community/site/README.md). Publish
allowlisted summaries, coverage and immutable evidence; keep traces, full datasets,
connection details and credentials private. Do not create Repairs entries for
unattributed numerical differences or diagnostic policy candidates.

## 15 Method references

The task details and tool contracts are owned by the repository recipes linked
above. The following primary references explain why the protocol separates
native numerical differences, repeatability and backend selection:

- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

The input counts, length bins and staged training/performance counts in this
runbook are project starting designs. Freeze actual scope per model and justify
any claim of statistical or application-level sufficiency using measured evidence.
