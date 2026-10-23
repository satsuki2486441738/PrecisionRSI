"""Plan bounded msProbe L0 forward capture without importing Torch or msProbe.
Call: plan_modules(model, module_paths, output_bytes, dump_path, max_tensor_bytes=None)
Input: Native module inventory, selected module paths and output byte estimates.
Output: config, observations, tensor_bytes for a bounded msProbe L0 plan.
Requires: Standard library; one invocation/module/step/rank; payload includes direct parameters. Verify capture counts/sizes in a trial.

Details:
API: plan_modules(model, module_paths, output_bytes, dump_path, rank=0, step=0,
                  max_tensor_bytes=None) -> {config, observations, tensor_bytes}.
model provides named_modules(); modules provide parameters(recurse=False).
output_bytes maps every selected native module path to the total bytes of its
expected output tensors, obtained from a small trial or known shapes/dtypes.

The plan covers one invocation per selected module in one step/rank. It emits
complete Module.<path>.<class>.forward.0 selectors and verifies their matches
against the model inventory, including numbered-prefix and descendant paths.
msProbe 26.1.0.post1 appends direct parameters even for forward/output filtering;
the payload estimate includes them. It excludes logs, allocator/serialization
and metadata overhead. Verify actual invocation counts, dump records and disk
usage in the trial; repeated calls or other modes need a task-owned plan.

This helper only builds configuration; it neither starts hooks nor changes
execution. Native capture metadata and semantic mappings remain adapter-owned.
"""
from pathlib import Path


def plan_modules(model, module_paths, output_bytes, dump_path, rank=0, step=0,
                 max_tensor_bytes=None):
    paths = list(module_paths)
    if not paths or any(not isinstance(p, str) or not p for p in paths) or len(set(paths)) != len(paths):
        raise ValueError('select unique nonempty native module paths')
    if type(rank) is not int or rank < 0 or type(step) is not int or step < 0:
        raise ValueError('rank and step must be nonnegative integers')
    if not isinstance(output_bytes, dict) or set(output_bytes) != set(paths):
        raise ValueError('output byte estimates must cover exactly the selected modules')
    if any(type(n) is not int or n <= 0 for n in output_bytes.values()):
        raise ValueError('each module needs a positive output byte estimate')
    if max_tensor_bytes is not None and (type(max_tensor_bytes) is not int or max_tensor_bytes <= 0):
        raise ValueError('tensor budget must be a positive integer')
    modules = dict(model.named_modules())
    if not set(paths) <= set(modules):
        raise ValueError('selected module missing from native model')
    names = {path: f'Module.{path}.{type(module).__name__}.forward.0'
             for path, module in modules.items()}
    selectors = [names[path] for path in paths]
    matches = {path for path, name in names.items() if any(s in name for s in selectors)}
    if matches != set(paths):
        raise ValueError('module selectors match unintended native modules')
    observations = []
    for path in paths:
        parameter_bytes = sum(p.numel() * p.element_size()
                              for p in modules[path].parameters(recurse=False))
        observations.append(dict(path=path, selector=names[path], invocation=0,
                                 output_bytes=output_bytes[path], parameter_bytes=parameter_bytes))
    total = sum(row['output_bytes'] + row['parameter_bytes'] for row in observations)
    if max_tensor_bytes is not None and total > max_tensor_bytes:
        raise ValueError(f'estimated tensor payload {total} exceeds budget {max_tensor_bytes}')
    return dict(config=dict(task='tensor', dump_path=str(Path(dump_path)), rank=[rank],
                            step=[step], level='L0', async_dump=False,
                            tensor=dict(scope=[], list=selectors, data_mode=['forward', 'output'])),
                observations=observations, tensor_bytes=total)
