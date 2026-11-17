# Bounded native NPU physical-cache supplement

Snapshot: 2026-11-10. The previously connection-blocked NPU probe has executed, passed source-bound and independent raw replays, and ended cleanly. This is a bounded supplemental result; overall scientific coverage remains incomplete. Numerical acceptance is unspecified.

## Design and observed state

One frozen512-token history ran in one native TP8/EP8 Engine: full-control, full-captured,253+259-control and253+259-captured. Original weights and production BF16-expanded/offload48/eager settings were reused. No completed277/216/adapted275 capture was rerun. Actual all8rank inputs/partitions were recorded. The actual nonpacked ABI uses32 FP32 state rows, not the GPU two-slot layout. Selected layers are2,8,14,20,24,28,32,36; ratio2 math producers are2,8,14 and source-alias producers are2,8,14,20.

All2304 retained ring-row writes exactly match the actual final32 FP32kv+score operands. The253 boundary reads preceding token252 at slot28;24 dependent first-carry capsules cover3sources and8TP ranks. All96 compressor/indexer aliases pass original tensor-identity checks. The768 declared cache-plane read reports contain280904 current-request initialized-row references and no unobserved declared-domain rows. These declarations are not an opaque kernel per-access trace.

The16 rank traces retain4848 raw NPY arrays (395786832file bytes). Each rank reserves49550922bytes across both captures, below128MiB. All128 dependent full129280-vocabulary head rows were normalized in FP64. Control/captured head arrays are byte-exact within each partition configuration. TP replicas and overlapping coordinates are not independent samples.

## Descriptive mathematics

The360 selected pool/norm capsules use actual native operands. FP64 pooling/native maximum-absolute-error range is 0.00042367588307257176 to 0.0018121966500133091; maximum RMSE is 0.00023569263824266138. RMSNorm recomputed from actual native pooled values and actual weight/epsilon has maximum-absolute-error range 0.002603985586482116 to 0.0077907431361254886, maximum RMSE 0.0012075962960744042. Projection and RoPE are not independently recomputed.

Full versus253+259 final-four logits differ: RMSE 0.40597446116146962, maxabs 2.4140625,517120 coordinates. The fixed input history matches while partitions differ. Full-vocabulary probability errors and top100-union/merged-tail JS are distinct summary fields; no top-k renormalization is applied. Differences alone do not establish a defect, acceptance, physical-cache equivalence or hardware-only cause.

## Integrity and cleanup

The5185 original files (685891226logical bytes) were exported losslessly as776 unique objects, with exact original-host before/after manifests. Independent fresh-emptycwd replay passes5203checks and exactly matches the local first review. Original-host/local FP64 reviews have216 low-bit differences among2288 floating fields, maxabs8.881784197001252e-16;18951 other fields are exact. Differences are retained.

The native task completed0. Shutdown returned normally within its60second budget. All100 attested process lifetimes ended; no cleanup signals were needed and authorized NPU devices were idle. This is not host-wide PID absence.

## Scope and history

OriginalGPU191/275, originalNPU269/275 and separate adaptedpaired275/275 remain distinct and unchanged. Twelve scientific gap entries are byte-preserved in the prior snapshot, cited by source ID/SHA. A512 probe does not establish32K physical carry, all-history/all40 mathematics, native own scores, full-backbone training or general cache equivalence. Output-head95 is not backbone training. Historical Core77/Core117 failures and the earlier connection-blocked snapshot retain their bytes; this newer evidence supersedes only the pending connection/probe status.

English/Chinese indexes and SVG diagnostics accompany this report. The writing package verifies exported integrity and numeric coordinates; full raw numerical replay is bound to private immutable receipts and hashes in the [canonical metrics](../../community/site/data/deepseek41-npu512-physical-cache-20261110.json). Automation remains user-paused. No push/publication occurred.
