# Layerwise comparison

Use after the user confirms investigating internal observations from a model-level comparison. Agree on samples, boundaries, execution stages and resource limits. Continue within that scope and record coverage as sampling expands.

## Collection

- Start with a few corresponding block boundaries; refine selected intervals into submodules or APIs. Match computational meaning, not just names, shapes or call order.
- Replay the model-level inputs and state through each native implementation. Intermediate differences include upstream propagation; identical-input replay of an isolated layer is a separately selected experiment.
- Identify each observation by sample/token coordinates, step/microbatch, semantic boundary, native module path, invocation, direction, input/output tensor path, rank and shard/view. Preserve dtype, shape, layout and selected slices; snapshot values at the capture boundary before later mutation.
- Use statistics for screening. Compute paired tensor metrics from actual values at matching coordinates. Record instrumentation changes to compilation, fusion or caching and use the [tensor definitions](comparison.md#tensors) for comparison. Report sampled boundaries, metrics, mappings, uncovered regions and candidate intervals for further sampling.

## msProbe quick reference

Use this summary to decide whether msProbe fits the selected observations. Reuse its public interfaces when suitable; native outputs or task-owned hooks are equally valid collection paths.

| Need | Available capability |
|---|---|
| Capture granularity | `L0` modules, `L1` APIs, or `mix` |
| Observations | Forward inputs/outputs and parameters; backward input/output gradients and parameter gradients, including nested tensor paths |
| Data volume | `statistics` for summaries; `tensor` for actual arrays; selected tensor capture can supplement statistics |
| Selection | `step`, `rank`, `scope`/`list`, `data_mode` for forward/backward and input/output; slicing where supported |
| Entry points | `msprobe.pytorch.PrecisionDebugger(config_path=...)`, `start(model=...)`, `stop()`, `step()` |
| Comparison | Module/API matching and tensor-error reports; adapt matching to the chosen cross-implementation boundaries |

Parameter-gradient counters count `grad_fn` triggers, not module calls. Map them to the task's accumulation/synchronization boundaries. Keep the task's random inputs and numerical policy when integrating collection helpers.

Check the installed version and execution mode before using an interface. Consult only the relevant [capture](https://msprobe.readthedocs.io/zh-cn/latest/zh/user_guide/dump/pytorch_data_dump_instruct.html), [configuration](https://msprobe.readthedocs.io/zh-cn/latest/zh/user_guide/dump/config_json_introduct.html) or [comparison](https://msprobe.readthedocs.io/zh-cn/latest/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.html) section for option details or an unsupported path; inspect source only as needed.
