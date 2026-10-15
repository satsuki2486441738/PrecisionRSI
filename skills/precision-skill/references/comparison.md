# Comparison, evidence, and handoff

## Frozen contract

Compare native end-to-end stacks with common inputs, shared implementation releases, and agreed numerical policies. Platform kernels, fusion, and internal precision can differ. Match parameter storage dtype, quantization, forward autocast policy, and scoring semantics; express equivalent settings through each platform's API.

Freeze the question and mode, required capabilities, observation points, complete or sampled scope, selection rules, source identities, model/data inputs, target semantics, initial state, randomness, mappings, reference direction, and metrics/thresholds. Keep platform-specific settings in side records. Apply [environment](environment.md) before new collection or historical reuse.

Layout differences require a verifiable transpose, reorder, shard reconstruction, or parameter mapping derived from semantics rather than fitted to observed results. A common subrange can be reported with explicitly reduced coverage.

Bind thresholds to the metric, reference, scope, aggregation, and comparison mode before the target comparison. Independent baseline experiments can characterize noise; do not derive acceptance criteria from the difference being judged.

## Schema-v2 scoring handoff

The loop is shared conditions, accepted single-side artifacts, counterpart execution, then offline comparison. Keep the contract and side manifests in the case; save execution records in its `runs/` directory. Create a counterpart manifest when its evidence exists. Keep the time when conditions were frozen distinct from the time of existing observations.

Both contract and manifests use integer `schema_version: 2`. The [comparator module docstring](../scripts/compare_scores.py) is the complete field/type contract.

| Object | Contents |
|---|---|
| Contract | Model/weights/tokenizer/config identities; common inputs; parameter dtype, quantization and autocast; eval/no_grad, full forward, cache off; FP32 natural-log scores; assistant-body nonspecial targets at t predicted from t−1; GPU reference; token-weighted NLL, MAE, global P99; coverage; `thresholds: null`. |
| Side manifest | Side, input/observation/summary/environment/status paths, effective conditions and evidence, expected ranks, execution state, uncovered items, and side-specific settings. |

Resolve resource paths relative to their containing JSON. Local paths locate resources; repository and revision establish source identity. Moving a handoff need not copy weights, environments, or tokenize the inputs again.

`actual` and `evidence` contain exactly these 15 fields:

```text
model, weights, tokenizer, config, weight_dtype, autocast, quantization,
training, no_grad, forward, use_cache, logprob_dtype, log_base,
target_scope, distribution
```

The four identities are nonempty `{repository, revision}` objects. Unknown facts remain null. Required missing evidence blocks coverage. Keep internal state dtype and a specific SDPA kernel as separate coverage items unless the task explicitly requires them; they are not mandatory common fields for v2 full-forward scoring.

Each `evidence.<field>` is a list of `{path, pointer}` objects. The pointer resolves inside an actual execution record to a value exactly equal to `actual.<field>`, including types and the complete key set. The contract or manifest cannot certify itself. The mechanism checks field consistency; provenance and evidence roles are reviewed by the collector/coordinator.

`autocast` has the exact shape:

```json
{"enabled": false, "target_dtype": null, "scope": "model_forward"}
```

`enabled` is a JSON boolean. When enabled, `target_dtype` is `float32`, `float16`, or `bfloat16`; when disabled it is null. The collector checks framework support for the chosen policy. Each side records the complete effective object.

The contract fixes `target_shift: -1`, `reduction: "target_token_weighted"`, and `percentile: "global_linear_0.99"`. These are core-owned rules and are excluded from `actual/evidence`. The core validates their declarations and executes array/metric checks. Report `rules` distinguish `definition`, `declaration_valid`, and `executed_at`; an empty execution list means the rule was not executed.

## Bundled comparator

The standard-library implementation imports neither Torch nor a model executor.

- API: `compare(contract_path, reference_manifest_path, candidate_manifest_path)` returns a structured report.
- CLI: `--contract --reference --candidate --output` exclusively creates the output. Exit codes are 0 for an available diagnostic, 1 for a written blocked report, and 2 for an argument/output error.
- Schema v2 / core 2.0.0 accepts `thresholds: null`. It implements full-forward scoring; cache, SFT, RL, tensor relative L2, inventory-derived facts, and a general policy DSL require separate task-specific tooling. It rejects v1 and the old `compute_dtype` field.
- Pair by `(source_row_1based, prompt_id, target_position)`. Validate complete histories, input/position IDs, attention/score masks, target IDs, shift, array lengths, uniqueness, and missing/extra entries. Reordering is supported; silently intersecting sets is not.
- Validate individual logprobs as finite and at most zero. Zero is valid; positive scores have no epsilon exemption. Recompute counts and summaries rather than trusting `finite: true`.
- Use Python double precision, fixed coordinate sorting, `math.fsum`, and global linear P99. Only summary rounding uses `rel_tol = abs_tol = 64 * sys.float_info.epsilon`; that tolerance is not a precision acceptance threshold.
- Check every expected rank's environment and completed status. Preserve execution failures and every concurrent blocker.
- Keep recomputable single-side diagnostics in `side_metrics`. Emit paired `metrics` only for complete comparable coverage. Invalid logprobs also block the affected side's NLL. Self-comparison does not waive unknown conditions.

Run from the skill directory, substituting actual paths:

```bash
python scripts/compare_scores.py \
  --contract '/work/case/contract.json' \
  --reference '/work/case/gpu.json' \
  --candidate '/work/case/npu.json' \
  --output '/work/case/runs/compare-scores-<UTC>/report.json'
```

## Evidence checklist

Collectors provide facts; the core checks supported relationships and metrics; the agent coordinates provenance review and gap investigation.

| Check | Record and validation | Missing-evidence action |
|---|---|---|
| Loaded identities | Model/weights/tokenizer/config `actual` and exact pointers to loaded `{repository, revision}` | Inspect loading records; requested paths alone do not identify actual loads. |
| Input and target coverage | Saved histories, positions, masks, targets, selection/body-boundary evidence | Replay arrays and validate relationships; reuse tokenization records. |
| Mode and quantization | Actual training/no_grad/forward/cache/quantization evidence | Look up execution context; eval mode alone does not establish no_grad. |
| Precision stages | Storage inventory, actual autocast, scoring dtype | Preserve each record's scope; investigate or adapt unsupported expressions. |
| Scoring semantics | Effective log base, target scope, and raw-model distribution | Resolve conflicts rather than changing the scoring definition. |
| Shift and reduction | Contract and actual arrays; per-token scores and summary | Revalidate t−1 and recompute token-weighted NLL. |
| P99 | Contract plus complete paired scores | Core computes the whole-set percentile; missing required inputs leave it unexecuted. |
| Completion and finite values | Observations, summary, rank environment/status, expected ranks | Retain missing coverage, failure stage, and any partial artifacts. |
| Shared versions | Run inventory or preparation/launch association | Follow the bounded alternatives in [environment](environment.md). |

Search the agreed record locations, then relevant existing logs/inventories within a bounded scope. Classify a gap as missing facts, unsupported representation, conflicting evidence, or unclear responsibility. Preserve candidate evidence when the interface cannot express it. A conflict retains both sources. Do not infer a rerun from null alone or substitute an agent's “verified” statement for a core-required value.

## Precision by stage

Separate intended configuration, implementation behavior, runtime observation, and unknowns. Inspect actual loaded configuration first; loaders can retain fields absent from a configuration class. Internal tensor dtype requires an appropriate runtime observation when it is necessary to the question.

| Stage | Evidence scope |
|---|---|
| Parameter/input storage | Parameter, buffer, and input dtypes for the inventoried objects. |
| Autocast and forward | Effective autocast enablement, target dtype, scope, and observed output/accumulation behavior where required. |
| Scoring | Logit origin, conversion location, log-softmax/gather path, and their observed dtypes. |
| Persistence and offline calculation | Conversion before saving, file representation, and comparator arithmetic. |

BF16 parameters can coexist with FP32 scoring. Storage dtype, autocast, or saved FP32 outputs do not establish the precision of every internal operation. Collect stages needed by the question rather than dumping the full graph. A required unknown remains a blocker; other internal details remain coverage limits. Record backend numerical switches, fusion, and known exceptions in side/environment records.

## Metrics

### Valid paired set

Let V contain all valid paired target coordinates and N = |V|. Check case identity, history, position, target ID, and mask. Exclude agreed padding/ignore positions and mask sentinels according to the contract. Missing observations are missing coverage; semantic, state, and mapping conflicts are incomparable. Empty sets and nonfinite valid values or metric results are incomparable. Preserve offending coordinates and values.

### Scores

Let l(d,i) be the natural log probability for the same target token. GPU is the default reference.

- `NLL_d = -sum(l(d,i)) / N`, in nat/token.
- `delta_NLL = NLL_NPU - NLL_GPU`. Sum across cases/batches/ranks before dividing by the total target count.
- `e_i = abs(l(NPU,i) - l(GPU,i))`; `MAE = sum(e_i) / N`, in nat.
- Sort all errors as `e[0] ... e[N-1]`. With `h = 0.99*(N-1)`, `j = floor(h)`, and `k = ceil(h)`, `P99 = e[j] + (h-j)*(e[k]-e[j])`.
- At N=1, P99 is the sole error. Report N, interpolation, and implementation version.

Signed delta NLL can conceal cancelling local differences, so also report absolute-error metrics. A negative delta is not an acceptance rule. Freeze a tolerated interval or absolute-value rule when acceptance is required. Sample percentiles describe their sampled set; perplexity need not be a separate acceptance metric.

### Tensors, gradients, and actual updates

For reference r and candidate c at the same coordinates, `relative_L2 = norm(c-r, 2) / norm(r, 2)`. Record tensor identity, flatten/concatenation scope, computation dtype, and both norms. Display near-zero reference norms explicitly.

| Reference norm | Error norm | Result |
|---|---|---|
| Positive | Finite | Apply the formula. |
| Zero | Zero | 0. |
| Zero | Positive | Undefined/incomparable; report the absolute error norm. |

Do not add epsilon to the denominator. Empty or nonfinite sets follow the valid-set rules. Compare gradients at the same scaling, accumulation, communication, and clipping stage. Compute actual updates separately on each side as `after - before`, then compare the two updates. Parameter closeness does not replace update comparison. Preserve the shared coordinate selection and counts for sharded or sampled tensors.

## Reporting and self-checks

| Status | Meaning |
|---|---|
| `passed` | Complete required evidence and coverage, comparable finite/defined metrics, frozen thresholds met. |
| `exceeded` | Comparable defined metric violates its frozen threshold. |
| `incomparable` | Semantic/state/mapping conflict or an undefined/nonfinite metric domain. |
| `not_covered` | Missing capability, execution, required observation, or range. |
| `no_thresholds` | Comparable numerical diagnostic without a matching prior threshold. |
| `execution_failed` | Attempted execution, collection, or calculation did not complete. |

Record execution, comparability, numerical state, coverage, and all blockers separately. Overall acceptance needs every required item and its required coverage to pass. The current core emits diagnostic/blocked states, not `passed` or `exceeded`; its precedence is execution_failed, incomparable, not_covered, then no_thresholds. A concurrent missing-threshold blocker remains visible.

Accept single-side inputs, identities, conditions, shift/masks, counts, finite observations, recomputed summaries, ranks, and references before pairing sides. The core fixes GPU as reference: GPU-to-GPU self-comparison checks its own handoff; GPU-to-NPU compares sides. Validate NPU artifacts through single-side checks before pairing with GPU. Zero self-difference is not cross-device evidence. Use small known differences to test formulas, direction, and domain handling; rerun checks when relevant inputs or implementation change.
