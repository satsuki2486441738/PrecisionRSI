# Compare artifacts

Use the existing Python environment with NumPy. Select one tool:

| Script | Comparison |
|---|---|
| [compare_distributions.py](../scripts/compare_distributions.py) | Full-vocabulary logprobs: top-100 union/tail metrics |
| [compare_tensors.py](../scripts/compare_tensors.py) | Gradients or stored before/after updates |
| [compare_scores.py](../scripts/compare_scores.py) | Cache-off eval target scores: NLL/MAE/P99; standard library |
| Task adapter | Native training scores/objectives; reuse the collector's mapping |
| [validate_artifacts.py](../scripts/validate_artifacts.py) | One platform's distribution/tensor inventory and values |

Run from the skill directory. Comparators share `--contract`, `--reference`, `--candidate` and `--output`; the validator uses `--manifest` for one side's input. Use fresh output paths.

```sh
python scripts/compare_tensors.py --contract gradients.json --reference reference.json --candidate candidate.json --output report.json
```

Use `--help` for CLI usage. When adapting inputs, extract only the selected module docstring with Python `ast`; distribution/tensor schemas share [artifact_io.py](../scripts/artifact_io.py). Inspect implementation functions as needed.

Contracts set `thresholds: null`. Outputs contain metrics, coordinate counts and unavailable comparisons. Exit codes: 0=result, 1=unavailable comparison, 2=CLI/output error.
