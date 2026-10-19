# Layerwise comparison

Use after the user confirms investigating internal observations from a model-level comparison. Infer samples, initial boundaries and execution stages from the task and existing comparison; confirm the scope and resource limits with the user. Within that authorization, choose tools, write adapters, run trials and refine sampling autonomously. Ask again only to change the comparison question, exceed resource limits, or resolve a material semantic ambiguity that code inspection and bounded trials cannot settle; provide the evidence and consequences needed to decide.

## Collection

- Start with a few corresponding block boundaries; refine selected intervals into submodules or APIs. Match computational meaning, not just names, shapes or call order.
- Replay the model-level inputs and state through each native implementation. Intermediate differences include upstream propagation; identical-input replay of an isolated layer is a separately selected experiment.
- Identify each observation by sample/token coordinates, step/microbatch, semantic boundary, native module path, invocation, direction, input/output tensor path, rank and shard/view. Preserve dtype, shape, layout and selected slices; snapshot values at the capture boundary before later mutation.
- Use statistics for screening. Compute paired tensor metrics from actual values at matching coordinates. Record instrumentation changes to compilation, fusion or caching and use the [tensor definitions](comparison.md#tensors) for comparison. Report sampled boundaries, metrics, mappings, uncovered regions and candidate intervals for further sampling.

## msProbe quick reference

Use this summary to decide whether msProbe fits the selected observations. Prefer its public capture and comparison interfaces when they cover the task. Use native outputs or task-owned hooks for other paths; different sides may use different collectors.

| Need | Available capability |
|---|---|
| Platforms | PyTorch collection on GPU and NPU; check the installed release for other backends and execution modes |
| Capture granularity | `L0` modules, `L1` APIs, or `mix` |
| Observations | Forward inputs/outputs and parameters; backward input/output gradients and parameter gradients, including nested tensor paths |
| Data volume | `statistics` for summaries; `tensor` for actual arrays; selected tensor capture can supplement statistics |
| Selection | `step`, `rank`, `scope`/`list`, `data_mode` for forward/backward and input/output; slicing where supported |
| Entry points | `msprobe.pytorch.PrecisionDebugger(config_path=...)`, `start(model=...)`, `stop()`, `step()` |
| Gaps in hooks | `PrecisionDebugger.save` for selected variables; check the release-specific signature and backward options |
| Comparison | Module/API matching and tensor-error reports; adapt matching to the chosen cross-implementation boundaries |

Parameter-gradient counters count `grad_fn` triggers, not module calls. Map them to the task's accumulation/synchronization boundaries. Keep the task's random inputs and numerical policy when integrating collection helpers.

## Integrate and verify

1. Inspect each side's framework, device and eager/compiled execution mode. Reuse msProbe if installed; otherwise select and pin a compatible `mindstudio-probe` release using the [installation guide](https://msprobe.readthedocs.io/zh-cn/latest/zh/install_guide/msprobe_install_guide.html). Constrain dependency resolution to the selected framework/runtime and numerical dependencies; test imports of the actual model class and collector before allocating devices. Use another collector if integration requires changing the compared implementation.
2. Keep configuration and adapter code with the task. Select `task` (`statistics` or `tensor`), `dump_path`, `level`, explicit `step` and `rank`, and the task-specific `scope`/`list` and `data_mode`. For one-call L0 observations, use [plan_modules](../scripts/msprobe_capture.py) with trial-derived output sizes to generate complete selectors and estimate tensor payload, including direct parameters. L0 list entries use substring matching; check the selected module set and actual disk use in the trial.
3. Construct `PrecisionDebugger(config_path=...)` in each executing worker early enough to instrument the selected calls. Bracket the native execution with `start(model=...)` and `stop()`; call `step()` after the captured iteration to finish writes and advance the capture step. Preserve the task's RNG and numerical settings rather than copying `seed_all()` from tutorials. Map debugger steps and gradient callbacks to the actual microbatch/accumulation stages.
4. Trial a small portion of the approved workload. Verify expected boundaries fired, per-rank files completed, and shapes, invocations and coordinates match the planned observations. Compare model-level outputs with collection disabled and enabled from the same initial state. Record instrumentation effects; adapt or replace collection when it changes the execution being compared.
5. Build explicit semantic mappings for the selected boundaries. Reuse msProbe comparison where its pairing and metric definitions fit; otherwise export selected actual tensors as `stage: observations` through the [comparison interfaces](comparison.md). Each tensor carries the task's matching logical `coordinates`; retain native capture metadata alongside it. Preserve source paths and the mapping from native records to logical coordinates, including unmatched observations and per-rank coverage.

Refine intervals within the approved budget using observed differences and coverage. Fix collector integration failures and retry autonomously; report sampled differences and unresolved coverage when the scope or budget is exhausted.

Consult only the relevant [capture](https://msprobe.readthedocs.io/zh-cn/latest/zh/user_guide/dump/pytorch_data_dump_instruct.html), [configuration](https://msprobe.readthedocs.io/zh-cn/latest/zh/user_guide/dump/config_json_introduct.html) or [comparison](https://msprobe.readthedocs.io/zh-cn/latest/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.html) section for option details or an unsupported path; use the [single-point save guide](https://msprobe.readthedocs.io/zh-cn/latest/zh/user_guide/dump/debugger_save_instruct.html) for uncovered variables. Inspect source only as needed.
