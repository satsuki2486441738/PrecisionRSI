# Offline tools

Run with the existing project Python environment. New artifact tools require NumPy; they import no model, Torch or device runtime. The legacy target-score comparator remains standard-library-only.

| Entry | Purpose |
|---|---|
| [validate_artifacts.py](../scripts/validate_artifacts.py) | Validate one side's complete expected positions/tensors, rank evidence and numerical values. |
| [compare_distributions.py](../scripts/compare_distributions.py) | Full-logprob top-100 union/tail metrics and coverage. |
| [compare_tensors.py](../scripts/compare_tensors.py) | Global/per-parameter gradient or actual-update metrics, streamed across differing shard layouts. |
| [compare_scores.py](../scripts/compare_scores.py) | Existing schema-v2 full-forward target-token NLL/MAE/P99. |

New tools use schema 1, separate from legacy target scoring. A thin case adapter writes a contract with reviewed common conditions and expected coverage, plus each side's manifest referencing existing arrays and execution evidence. Reuse files; do not copy large tensors into the index. NumPy and safetensors float arrays, including BF16 parameters, are supported. Schemas are defined in the entry scripts and [artifact_io.py](../scripts/artifact_io.py), not duplicated here.

```bash
python scripts/validate_artifacts.py --contract contract.json --manifest gpu.json --output gpu-audit.json
python scripts/compare_tensors.py --contract gradients.json --reference gpu.json --candidate npu.json --output gradients-report.json
python scripts/compare_distributions.py --contract distributions.json --reference gpu.json --candidate npu.json --output distribution-report.json
```

Run from the skill directory. Inputs resolve relative to their manifest; outputs are exclusively created. Exit codes: 0 for a valid audit/diagnostic, 1 for a blocked scope, 2 for CLI/output errors. Pairing uses a GPU reference and NPU candidate; using the same manifest twice produces an explicitly labeled self-check. No tool emits an acceptance pass or chooses thresholds.

Tensor updates accept stored before/after values or FP32/FP64 deltas. Before/after pairs also permit exact initial-value checks; delta-only files rely on reviewed initial-state provenance. Zero norms and skipped steps retain valid metric scopes. Missing data blocks complete aggregates; valid per-position/per-tensor diagnostics remain available.

These tools verify declared conditions and artifacts, not the truth of backend/provenance claims. Collectors still supply canonical vocabulary/parameter maps and native observations; tensor shards must map to contiguous flat logical ranges. SFT/RL action-score and objective reconstruction, model execution, cache-state comparison and engine stage/length breakdowns remain case work.
