# Qwen3.5 GPU–NPU precision delivery

This delivery indexes the Qwen3.5-9B-Base evidence in the 8 November 2026 paper-writing snapshot. Results are bounded by the model revision, input selection and execution settings in each report. The snapshot is an aggregate export; numerical acceptance is unspecified.

## Reports and selected findings

- [Qwen3.5 precision evaluation](paper-writing-results-20261108/reports/qwen35.md): In selected inference positions, repeated-text inputs had 264/264 top-1 matches and the separate structured selection had 45/48. Native NPU deterministic level 1 restored the report's selected repeats and five-step checkpoint continuation. This is a configuration-specific reproducibility result, not cross-platform equivalence.
- [Qwen diagnostics, interim](paper-writing-results-20261108/reports/qwen-diagnostics.md): The Qwen3.5 sections cover backward-boundary diagnosis, historical continuation, a nine-example heldout probe and GPU cache observations. The original primary-host NPU cache capture had not launched at this snapshot. The report also includes Qwen3.8 and DeepSeek results.
- [Qwen follow-up](paper-writing-results-20261108/reports/qwen-followup.md): A separately admitted secondary-host NPU cache cohort was compared with retained GPU observations across 144 causal rows, with first-max IDs agreeing on 135. This does not replace the missing primary-host capture. The report also includes Qwen3.8 results.
- [Same-input operator references and software-stack robustness](paper-writing-results-20261108/reports/operator.md): This related operator supplement covers 58 cases and 194 reference arrays. It distinguishes mathematical reference error, agreement and repeatability; alternate stacks change multiple components together and their coverage differs.

The [DeepSeek report](paper-writing-results-20261108/reports/deepseek41.md) is included as shared context because the unchanged Qwen follow-up reports reference it. Its GPU mixed-FP8/FP4 and adapted NPU BF16 configurations also differ in parallel layout and offload; numerical gaps remain. The [figures](paper-writing-results-20261108/figures/) are browsable copies of the referenced originals.

## Original evidence and verification

The [complete original ZIP](paper-writing-results-20261108.zip) is a shared cross-model archive, not a Qwen3.5-only dataset. It includes Qwen3.5, Qwen3.8, DeepSeek and operator reports, source JSON files, `results.json`, `values.csv`, figures, a SHA-256 manifest and `verify.py`. Reports and figures included outside the ZIP are byte-identical browsable copies; links from them to source data require the extracted archive.

Extract the archive and run `python -I -B verify.py` in the extracted directory to check the supplied bytes against the manifest. This checks integrity, not experimental replay. The ZIP SHA-256 is `f66fdb64f996087bac2761a9d25a6d922df655c66c433351427c96ecee14d54f`. A separate import check found 23 embedded sources consistent with their exported JSON and all 74,722 CSV numeric entries consistent with their source pointers.

Private raw arrays, model weights, full datasets and private replay tools are not included. Missing coverage is not a pass; GPU is not a mathematical oracle. Independent examples, dependent coordinates, repeated processes and overlapping prefixes are distinct units of observation.
