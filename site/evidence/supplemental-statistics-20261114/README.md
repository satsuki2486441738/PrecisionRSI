# Supplemental historical precision statistics

This [statistics-only bundle](statistics.json) contains 13 historical evaluation
snapshots: four Qwen3.8 and nine DeepSeek records, 40 comparisons and 101 measured
metrics. Its source catalog snapshot is dated 2026-11-09. The export date is
2026-11-14; it is not a new experiment or an update to those historical outcomes.

Three records are historically completed and ten partial. All 13 retain unknown
numerical acceptance (`null`). Each record retains its model revision, peer
configuration, metrics, coverage, comparison scope, limitations and original
observation date. The confirmed hardware display name is H200. Original private
source bytes are unchanged; the source catalog and each complete original record
are bound by SHA-256 fingerprints.

No raw arrays, prompts, responses, execution traces, private source paths or
source archive URLs are supplied. The public bundle does not provide independent
numerical replay. References to portable replay or archive verification in the
historical notes describe the retained source evidence, not a downloadable
feature of this bundle. Source Git revision public reachability is not asserted.

The export was checked field by field against its source: only hardware display
strings changed, while source `archive` and `evidence` metadata were replaced by
explicit statistics-only provenance. This checks faithful projection; it does
not recompute the original numerical experiments or establish acceptance.

| Evaluation ID | Historical status | Comparisons | Metrics |
| --- | --- | ---: | ---: |
| qwen38-native-inference-20261106 | completed | 4 | 12 |
| qwen38-native-lora-frameworks-20261106 | completed | 3 | 6 |
| qwen38-native-rl-components-20261106 | completed | 4 | 8 |
| qwen38-native-serving-admissions-20261106 | partial | 3 | 9 |
| deepseek41-original-vs-adapted-core-20261106 | partial | 1 | 4 |
| deepseek41-upstream-compressor-key-local-vjp-20261108 | partial | 2 | 4 |
| deepseek41-native-query-tail-20261108 | partial | 4 | 8 |
| deepseek41-adapted-core275-20261109 | partial | 2 | 4 |
| deepseek41-full40-attention-20261109 | partial | 1 | 2 |
| deepseek41-paired-output-head95-20261109 | partial | 1 | 3 |
| deepseek41-http128-20261109 | partial | 6 | 20 |
| deepseek41-native-producer-key-cache-20261109 | partial | 6 | 11 |
| deepseek41-native-query-key-selection-20261109 | partial | 3 | 10 |

## Interpretation boundaries

- Keep original GPU 191/275, original NPU 269/275 and separately adapted paired
  275/275 coverage distinct. These records do not retroactively close rejected
  original cache or padding APIs.
- Output-head LoRA updates are not backbone training. Component or selected
  query/cache observations do not establish full-model or full-service coverage.
- Historical incomplete service statements retain their original observation
  meaning. Later SGLang/vLLM experiments and prospective lifecycle/cache repairs
  require their own evidence and must not rewrite these snapshots.
- Fixed teacher histories, fresh-prefill execution and physical reference-cache
  equivalence are separate conditions. Diverged generation histories are unpaired.
- Full-vocabulary normalization and tail-merged distribution metrics retain their
  declared definitions; a merged-tail JS value is not full-coordinate JS.
- Comparison differences do not establish a confirmed implementation defect,
  hardware causality, task accuracy or numerical acceptance.

This is an evidence format separate from the live replay-backed evaluation
catalog. It changes no frontend routes or public history records. Future data
integration should expose the statistics-only scope explicitly.

The maintainer exporter is `.github/export_statistics.py`; its integration and
publication-review procedure is documented in
[HISTORY_INTEGRATION.md](../../../.github/HISTORY_INTEGRATION.md).
