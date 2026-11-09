# DeepSeek-V4.1 GPU–NPU precision delivery

This delivery indexes the `deepseek-ai/DeepSeek-V4.1-Flash-1017` evidence in the 8 November 2026 paper-writing snapshot. The model revision, inputs, configurations, coverage and limitations are recorded in the [primary report](paper-writing-results-20261108/reports/deepseek41.md). This is an aggregate export; numerical acceptance remains unspecified.

## Selected findings and scope

- The complete-model captures support a descriptive comparison of **191 of 275 expected tensor observations** and all **3,168 target log-probabilities**. Their mean absolute difference is 0.280315 nat. All six controlled candidate questions have the same first choice, while some token distributions and lower candidate ordering differ. These bounded probes do not establish general model accuracy.
- The GPU uses the original mixed-FP8/FP4 reference implementation with TP4; the NPU uses an adapted BF16 configuration with TP8/EP8 and 48 GB CPU offload. The comparison therefore includes arithmetic, parallel-layout and placement differences and does not isolate a hardware effect or establish equivalent backend arithmetic.
- The report preserves missing masked-padding coverage, GPU reference-cache rejections and numerical gaps in training and several framework chains. Missing coverage is not a pass. The [precision diagnostic figure](paper-writing-results-20261108/figures/3-deepseek41-precision-diagnostics.svg) is a byte-identical browsable copy of the original.

## Original evidence and verification

The [DeepSeek source JSON](paper-writing-results-20261108/sources/deepseek41-precision-20261106.json) is available alongside the report. The [complete original ZIP](paper-writing-results-20261108.zip) is a shared cross-model archive, not a DeepSeek-only dataset. It contains the original reports, figures, all source JSON files, `results.json`, `values.csv`, a SHA-256 manifest and `verify.py`, including Qwen and operator evidence.

Extract the ZIP and run `python -I -B verify.py` in the extracted directory to check the included files against the manifest. This verifies byte integrity only; it does not replay the experiments. The archive SHA-256 is `f66fdb64f996087bac2761a9d25a6d922df655c66c433351427c96ecee14d54f`. Private raw arrays, model weights, full datasets and private replay tools are not included.
