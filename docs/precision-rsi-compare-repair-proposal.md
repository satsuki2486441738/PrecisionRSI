# PrecisionRSI Compare–Repair Architecture Proposal

**Status:** Proposal for review
**Branch:** `proposal/precision-rsi-compare-repair-architecture`
**Base:** remote `main` at `57e1505411c0f0c2a5f4b00ed2f93f7b8b25f09e`
**Scope:** information architecture and data semantics only; no runtime, measurement, or public-result migration is included in this branch.

## Executive decision

PrecisionRSI should be presented as an open precision-engineering community with two equal first-class pillars:

1. **Precision Compare** measures and explains numerical differences between explicitly scoped hardware, software, model, workload, and configuration peers.
2. **Precision Repair** diagnoses implementation defects, records causal changes, and verifies fixes on the original workload.

A shared **Ecosystem Index** helps users find records by hardware, workflow, library, model, and setting. It is a discovery layer, not a third scientific claim and not a replacement for either pillar.

The public statement should be:

> PrecisionRSI is an open community for comparing numerical precision across AI hardware and software stacks, and for diagnosing, repairing, and verifying implementation defects.

## Semantic boundary

A comparison is evidence of an observed relationship under a declared scope. It does not automatically establish a defect, root cause, hardware fault, support guarantee, or numerical acceptance.

A repair record requires a separate causal and verification chain. It must identify the affected implementation, the trigger, the expected behavior, the observed behavior, the change, and the original-workload verification scope.

The transition is explicit:

```text
scoped configuration(s)
        └── Precision Compare record
                ├── observed differences and coverage
                └── optional repair candidate
                        └── Precision Repair record
                                ├── diagnosis / causal evidence
                                ├── patch or upstream change
                                └── original-workload regression
```

A comparison may have no repair. A repair may link multiple comparisons and verification rounds. An investigation remains an investigation until its evidence supports the stronger label.

## Proposed navigation

```text
PrecisionRSI
├── Precision Compare
├── Precision Repair
├── Ecosystem Index
└── Evidence & Research
```

The landing page should explain the two pillars before exposing the index:

- **Compare:** “How different are these environments under the same declared scope?”
- **Repair:** “Is the difference caused by an implementation defect, and has a fix been verified?”
- **Ecosystem Index:** “For this card and workload, which comparison and repair records should I inspect?”

The existing Differences view belongs under **Precision Compare**. The existing repair archive belongs under **Precision Repair**. The hardware × workflow × library board is a shared index that links to both record types.

## Page responsibilities

### Precision Compare

Primary filters:

- hardware vendor, device, architecture, and device type;
- runtime, compiler, framework, backend, operator, and communication components;
- model and revision;
- workload: inference, SFT, RL, component, or other declared task;
- dtype, quantization, TP/EP, batch, context, cache, graph/eager, and determinism settings;
- comparison status and evidence level.

A compare row should show the peer configurations, the metric and unit, observed/expected coverage, the comparison basis, independent-review state, and remaining limits. Labels such as “observed difference”, “independently replayed”, and “acceptance unset” are preferable to a generic “problem” badge.

### Precision Repair

Primary filters:

- affected implementation and library group;
- workflow and model;
- defect state: reported, reproduced, diagnosed, candidate fix, validated fix, or not reproduced;
- repair and upstream state;
- verification coverage and remaining limits.

A repair detail must link the comparison or reproduction evidence that motivated it, but it must retain its own cause, change, before/after metrics, original workload, and validation records.

### Ecosystem Index

The index is a read-only projection over both pillars. Its matrix can use:

```text
library / observed environment × workflow

cell:
  compare records: N
  repair records: M
  investigations: K
  verified configurations: V
```

A blank cell means “no public record”, never “supported” or “passed”. A comparison count must not be displayed as a defect count. A repair count must not be generalized to every version or setting.

## Shared data model

Keep the current immutable experiment and repair records. Add stable relationships rather than rewriting historical measurements:

```text
Hardware / Stack / Model / Workload / Settings
        ├── compare_record[]
        └── repair_issue[]
                └── verification_record[]
```

Each observed configuration should carry structured IDs where available and preserve the original display string. A stack is decomposed by role: driver/runtime, framework, engine, adaptation backend, compiler, operator/kernel, and collective/communication library. Unknown vendors, devices, ecosystems, and versions remain valid values.

Recommended cross-record fields:

- `record_type`: `comparison`, `repair`, or `verification`;
- `scope`: exact model revision, input/workload, task, and settings;
- `environment`: hardware and role-tagged software components with versions/commits;
- `evidence`: source pointers, immutable artifact hashes, reproduction recipe, and review level;
- `status`: collection/reproduction/diagnosis/verification state;
- `acceptance`: nullable and independent from completion status;
- `links`: upstream issue/PR, related comparisons, related repairs, and superseding records.

Presentation aliases such as `torch` → `PyTorch` may be used for indexing, but raw names and source bytes remain unchanged. A hardware/library filter must match the same recorded side or environment; do not form a false cross-side join from independent peer lists.

## Migration from the current repository

Start from the latest remote `main` and preserve its current survey Differences implementation, source-verified data, Gemma4 records, collector, validator, and tests.

Then integrate in separate, reviewable stages:

1. Add the shared record projection and Ecosystem Index without changing comparison or repair source records.
2. Add the two-pillar landing copy and navigation; keep the full Differences, evaluation, and repair views reachable by stable links.
3. Merge additional reviewed DeepSeek/Qwen records by stable ID, retaining original dates, source pointers, null acceptance, and explicit scope limits.
4. Add paper/results indexes only after public-artifact and privacy review; large evidence archives remain an explicit release decision.
5. Add CI checks for both existing Differences data and the new projection. Never remove `validate_differences.py` from the publish workflow.
6. Keep `precision_rsi` runtime changes, research snapshots, DSH overlays, and operator tooling in a separate PR from public-site information architecture.

No migration should delete a remote survey record or silently replace an older archive with a later snapshot. Conflicts are resolved by stable ID and provenance review, not by choosing the larger JSON file.

## Decisions required before implementation

1. **Landing page:** approve the two-pillar landing page with Ecosystem Index as the shared discovery view.
2. **Compare view:** retain the existing `main` Differences dashboard as the canonical Compare surface.
3. **Data merge:** decide whether the local DeepSeek/Qwen records are ready for public inclusion, and whether the remote Gemma4 records remain first-class.
4. **Evidence release:** approve publication of paper JSON/CSV and reports; decide separately for large ZIP archives and any raw-array-derived artifacts.
5. **Runtime scope:** keep RSI runtime/research infrastructure changes in a separate PR.
6. **Terminology:** approve “observed difference” for comparison output and reserve “defect/problem” for repair investigations with the corresponding evidence.

## Non-goals for this proposal branch

- changing numerical values, acceptance decisions, or historical dates;
- merging or deleting data archives;
- changing model services, accelerators, or experiment owners;
- publishing private raw arrays, host identifiers, credentials, or datasets;
- implementing the final UI or altering the runtime packages.
