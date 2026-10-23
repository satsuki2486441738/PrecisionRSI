"""Compare nested native state without prescribing a checkpoint format (NumPy).
Call: compare_trees(a, b, leaf_adapter=None, metrics=False); CLI: python scripts/tree_compare.py A.pt B.pt --output NEW.json [--metrics]
Input: Nested mappings/sequences with numeric/text leaves, or trusted Torch checkpoint files.
Output: Equality, structural/metadata differences and numeric coverage/findings; checkpoints add hashes/sections and original state paths. --all-findings returns all differences.
Requires: NumPy; leaf adapters retain native dtype. Checkpoints require Torch and load on CPU; numeric equality requires finite leaves.

Details:
compare_trees(a, b, *, leaf_adapter=None, metrics=False, max_findings=8)
accepts mappings with Python str/int keys, lists, tuples, real NumPy arrays and
scalars, Python numbers, text and None. Mapping order is ignored; key types,
sequence types, array shapes/dtypes and metadata are preserved. Shared subtrees
are allowed; cycles and unsupported values raise. Inputs are not modified.

leaf_adapter converts an unsupported framework object to supported values. It
can return {"dtype": native_dtype, "values": exported_array} to retain native
identity when export needs conversion. Loaders, subtree selection and semantic
alignment belong to the task adapter. The Python tree API imports no framework
and loads no files.

The report has status="completed", exact_equal, coverage, metadata and numeric.
coverage contains inventory="observed_union", side_a_nodes, side_b_nodes,
paired_nodes and missing_nodes counts by side. metadata contains differences,
findings (path/side_a/side_b descriptors) and findings_omitted. numeric is the
existing compare_arrays report, or None when no numeric leaves exist. Shared omissions need an independent task
inventory. Exact equality requires matching structure/metadata and finite,
exactly equal numeric leaves. metrics=True adds the existing FP64 metrics;
Python integers outside NumPy's integer range remain exact text metadata.

Paths contain typed key/index components, avoiding slash and integer/string key
collisions. Numeric names encode these paths as JSON; findings and maximum-error
coordinates also include decoded paths. Metadata and numeric findings are each
bounded by max_findings; aggregate counts retain all differences. Set
max_findings=None (CLI --all-findings) for every numeric difference/nonfinite
coordinate and every structural/metadata finding, with original scalar values.

For trusted Torch-saved state, CLI (optional Torch dependency):
python scripts/tree_compare.py A.pt B.pt --output NEW.json [--metrics]
                              [--max-findings 8 | --all-findings]
Loads both states onto CPU. The result contains inputs (paths/SHA-256), runtime
versions, comparison (the whole tree report) and sections (per top-level key
reports for mappings). Each numeric array is scanned once; section statistics
and the global report reuse those reductions and native dtype conversions.
Checkpoint paths refer directly to stored keys/indices, including tensor leaves;
no conversion-wrapper components are added. Native tensor dtypes are retained
in node descriptors and reported values. Signed zeros compare numerically equal.
Findings already include typed paths, indices and native
values; counts cover all supplied state even when examples are bounded.
No checkpoint field names or training layout are assumed. Other native formats
use compare_trees with their own loader. Exit 0=comparison completed, including
differences; 2=invalid input/output. Inspect comparison.exact_equal for equality.
"""
from collections.abc import Mapping
import json

import numpy as np

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from array_compare import compare_arrays, _compare_partitions


def _name(path):
    return json.dumps(path, ensure_ascii=False, separators=(',', ':'))


class _NativeArray:
    def __init__(self, dtype, values):
        self.dtype, self.values = dtype, values


def _flatten(value, leaf_adapter):
    nodes, arrays, active = {}, {}, set()

    def visit(value, path, allow_adapter=True):
        name = _name(path)
        if isinstance(value, (Mapping, list, tuple)):
            identity = id(value)
            if identity in active:
                raise ValueError('cyclic state at ' + name)
            active.add(identity)
            try:
                if isinstance(value, Mapping):
                    nodes[name] = dict(kind='mapping')
                    for key, child in value.items():
                        if type(key) not in (str, int):
                            raise TypeError('mapping keys must be strings/integers at ' + name)
                        visit(child, path + [['key', type(key).__name__, str(key)]], allow_adapter)
                else:
                    nodes[name] = dict(kind=type(value).__name__)
                    for index, child in enumerate(value):
                        visit(child, path + [['index', index]], allow_adapter)
            finally:
                active.remove(identity)
        elif isinstance(value, _NativeArray):
            if value.values.dtype.kind not in 'biuf':
                raise TypeError('real numeric arrays required at ' + name)
            nodes[name] = dict(kind='numeric', native_dtype=value.dtype)
            arrays[name] = value.values
        elif value is None or isinstance(value, str):
            nodes[name] = dict(kind='none' if value is None else 'str', value=value)
        elif isinstance(value, (np.ndarray, np.generic)) or type(value) in (bool, int, float):
            array = np.asarray(value)
            if type(value) is int and array.dtype.kind == 'O':
                nodes[name] = dict(kind='integer', value=str(value))
            else:
                if array.dtype.kind not in 'biuf':
                    raise TypeError('real numeric arrays required at ' + name)
                nodes[name] = dict(kind='numeric')
                arrays[name] = array
        elif leaf_adapter is not None and allow_adapter:
            identity = id(value)
            if identity in active:
                raise ValueError('cyclic adapted state at ' + name)
            active.add(identity)
            try:
                adapted = leaf_adapter(value)
                if adapted is value:
                    raise TypeError('leaf_adapter must return supported values at ' + name)
                visit(adapted, path, False)
            finally:
                active.remove(identity)
        else:
            if id(value) in active:
                raise ValueError('cyclic adapted state at ' + name)
            raise TypeError('unsupported state value at ' + name + ': ' + type(value).__name__)

    visit(value, [])
    return nodes, arrays


def compare_trees(a, b, *, leaf_adapter=None, metrics=False, max_findings=8):
    """Compare corresponding nested state; adapt only unsupported framework leaves."""
    if (leaf_adapter is not None and not callable(leaf_adapter) or
            type(metrics) is not bool or
            (max_findings is not None and (type(max_findings) is not int or max_findings < 0))):
        raise ValueError('invalid adapter, metrics or max_findings')
    states = [_flatten(value, leaf_adapter) for value in (a, b)]
    nodes, arrays = zip(*states)
    numeric = compare_arrays(*arrays, metrics=metrics, max_findings=max_findings) if any(arrays) else None
    return _tree_report(nodes, numeric, max_findings)


def _native_values(values, name, nodes):
    exported = values['dtype']
    dtypes = {side: nodes[index].get(name, {}).get('native_dtype') or
              (exported[side] if isinstance(exported, dict) else exported)
              for index, side in enumerate(('side_a', 'side_b'))}
    values['dtype'] = dtypes['side_a'] if dtypes['side_a'] == dtypes['side_b'] else dtypes


def _tree_report(nodes, numeric, max_findings):
    names = set(nodes[0]) | set(nodes[1])
    findings, differences = [], 0
    for name in sorted(names):
        values = [side.get(name) for side in nodes]
        if values[0] != values[1]:
            differences += 1
            if max_findings is None or len(findings) < max_findings:
                findings.append(dict(path=json.loads(name), side_a=values[0], side_b=values[1]))
    if numeric is not None:
        for finding in numeric['findings']:
            finding['path'] = json.loads(finding['name'])
            if 'values' in finding:
                _native_values(finding['values'], finding['name'], nodes)
            for index, side in enumerate(('side_a', 'side_b')):
                dtype = nodes[index].get(finding['name'], {}).get('native_dtype')
                if dtype and side + '_dtype' in finding:
                    finding[side + '_dtype'] = dtype
        coordinate = (numeric.get('metrics') or {}).get('max_coordinate')
        if coordinate is not None:
            coordinate['path'] = json.loads(coordinate['name'])
            _native_values(numeric['metrics']['max_values'], coordinate['name'], nodes)
    return dict(status='completed',
                exact_equal=differences == 0 and (numeric is None or numeric['exact_equal']),
                coverage=dict(inventory='observed_union', side_a_nodes=len(nodes[0]),
                              side_b_nodes=len(nodes[1]), paired_nodes=len(set(nodes[0]) & set(nodes[1])),
                              missing_nodes=dict(side_a=len(names-set(nodes[0])),
                                                 side_b=len(names-set(nodes[1])))),
                metadata=dict(differences=differences, findings=findings,
                              findings_omitted=differences-len(findings)), numeric=numeric)


def compare_checkpoints(side_a, side_b, *, metrics=False, max_findings=8):
    """Load Torch state files on CPU and return whole-tree and top-level reports."""
    import hashlib
    from pathlib import Path
    import torch

    def export(value):
        if not isinstance(value, torch.Tensor):
            raise TypeError('unsupported checkpoint leaf: ' + type(value).__name__)
        tensor = value.detach().cpu()
        values = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
        return _NativeArray(str(tensor.dtype), values)

    def source(path):
        path = Path(path).resolve()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        return dict(path=str(path), sha256=digest.hexdigest())

    inputs = {side: source(path) for side, path in (('side_a', side_a), ('side_b', side_b))}
    states = [torch.load(item['path'], map_location='cpu', weights_only=False) for item in inputs.values()]
    if (type(metrics) is not bool or
            (max_findings is not None and (type(max_findings) is not int or max_findings < 0))):
        raise ValueError('invalid metrics or max_findings')
    flattened = [_flatten(value, export) for value in states]
    nodes, arrays = zip(*flattened)
    sections, partitions, section_nodes = [], {}, {}
    if all(isinstance(state, Mapping) for state in states):
        for key in sorted(set(states[0]) | set(states[1]), key=lambda key: (type(key).__name__, str(key))):
            prefix = [['key', type(key).__name__, str(key)]]
            group = _name(prefix)
            section_nodes[group] = [{'[]': side['[]']} for side in nodes]
            partitions[group] = set()
            sections.append(dict(key=key, key_type=type(key).__name__))
        for index, side in enumerate(nodes):
            for name, value in side.items():
                if name != '[]':
                    group = _name(json.loads(name)[:1])
                    section_nodes[group][index][name] = value
                    if name in arrays[index]:
                        partitions[group].add(name)
    if any(arrays):
        numeric, grouped = _compare_partitions(*arrays, metrics=metrics, max_findings=max_findings,
                                               partitions=partitions)
    else:
        numeric, grouped = None, {}
    report = _tree_report(nodes, numeric, max_findings)
    for section, (group, selected) in zip(sections, section_nodes.items()):
        section['comparison'] = _tree_report(selected, grouped[group] if partitions[group] else None,
                                             max_findings)
    return dict(inputs=inputs, runtime=dict(torch=torch.__version__, numpy=np.__version__),
                comparison=report, sections=sections)


def main(argv=None):
    import argparse
    import pickle
    from pathlib import Path
    import sys
    parser = argparse.ArgumentParser(description=__doc__.partition('\n\nDetails:\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('side_a')
    parser.add_argument('side_b')
    parser.add_argument('--output', required=True)
    parser.add_argument('--metrics', action='store_true')
    findings = parser.add_mutually_exclusive_group()
    findings.add_argument('--max-findings', type=int, default=8)
    findings.add_argument('--all-findings', dest='max_findings', action='store_const', const=None,
                          help='return every difference with original state paths and native values')
    args = parser.parse_args(argv)
    try:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError('output already exists: ' + str(output))
        report = compare_checkpoints(args.side_a, args.side_b, metrics=args.metrics,
                                     max_findings=args.max_findings)
        with output.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return 0
    except (ImportError, OSError, ValueError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
