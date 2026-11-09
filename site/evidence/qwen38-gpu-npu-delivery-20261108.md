# Qwen3.8 GPU–NPU precision delivery

This delivery indexes the Qwen3.8-Flash-Next evidence in the 8 November 2026 paper-writing snapshot. Results are bounded by the model revision, input selection and execution settings in each report. The snapshot is an aggregate export; numerical acceptance is unspecified.

## Reports and selected findings

- [Qwen3.8 precision evaluation](paper-writing-results-20261108/reports/qwen38.md): For GPU automatic placement versus NPU TP8, selected full-forward positions matched at 254/264 top-1 positions and later fixed-history rows at 2/32. Different placement and parallel layouts are part of the comparison. Two NPU engine admissions failed before scoring, leaving numerical serving coverage missing.
- [Qwen diagnostics, interim](paper-writing-results-20261108/reports/qwen-diagnostics.md): In the selected original TP8 fixed-history workload, the packed-loader repair changed first-ID agreement from 2/32 to 31/32; residual differences remain. The report also includes Qwen3.5 and DeepSeek results.
- [Qwen follow-up](paper-writing-results-20261108/reports/qwen-followup.md): Its Qwen3.8 section reports a matched stock/fixed cost comparison under the recorded conditions. The observed order and configuration scope limit the cost interpretation. The report also includes Qwen3.5 results.

The [DeepSeek report](paper-writing-results-20261108/reports/deepseek41.md) is included as shared context because the unchanged Qwen follow-up reports reference it. Its GPU mixed-FP8/FP4 and adapted NPU BF16 configurations also differ in parallel layout and offload; numerical gaps remain. The [figures](paper-writing-results-20261108/figures/) are browsable copies of the referenced originals.

## Original evidence and verification

The [complete original ZIP](paper-writing-results-20261108.zip) is a shared cross-model archive, not a Qwen3.8-only dataset. It includes Qwen3.5, Qwen3.8, DeepSeek and operator reports, source JSON files, `results.json`, `values.csv`, figures, a SHA-256 manifest and `verify.py`. Reports and figures included outside the ZIP are byte-identical browsable copies; links from them to source data require the extracted archive.

Extract the archive and run `python -I -B verify.py` in the extracted directory to check the supplied bytes against the manifest. This checks integrity, not experimental replay. The ZIP SHA-256 is `f66fdb64f996087bac2761a9d25a6d922df655c66c433351427c96ecee14d54f`. A separate import check found 23 embedded sources consistent with their exported JSON and all 74,722 CSV numeric entries consistent with their source pointers.

Private raw arrays, model weights, full datasets and private replay tools are not included. Missing coverage is not a pass; GPU is not a mathematical oracle. Independent examples, dependent coordinates, repeated processes and overlapping prefixes are distinct units of observation.
