"""Compare nested native state without prescribing a checkpoint format (NumPy).

compare_trees(a, b, *, leaf_adapter=None, metrics=False, max_findings=8)
accepts mappings with Python str/int keys, lists, tuples, real NumPy arrays and
scalars, Python numbers, text and None. Mapping order is ignored; key types,
sequence types, array shapes/dtypes and metadata are preserved. Shared subtrees
are allowed; cycles and unsupported values raise. Inputs are not modified.

leaf_adapter converts an unsupported framework object to supported values. It
can return {"dtype": native_dtype, "values": exported_array} to retain native
identity when export needs conversion. Loaders, subtree selection and semantic
alignment belong to the task adapter. No framework import or file loading occurs.

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
bounded by max_findings; aggregate counts retain all differences.
"""
from collections.abc import Mapping
import json

import numpy as np

from array_compare import compare_arrays


def _name(path):
    return json.dumps(path, ensure_ascii=False, separators=(',', ':'))


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
            type(metrics) is not bool or type(max_findings) is not int or max_findings < 0):
        raise ValueError('invalid adapter, metrics or max_findings')
    states = [_flatten(value, leaf_adapter) for value in (a, b)]
    nodes, arrays = zip(*states)
    names = set(nodes[0]) | set(nodes[1])
    findings, differences = [], 0
    for name in sorted(names):
        values = [side.get(name) for side in nodes]
        if values[0] != values[1]:
            differences += 1
            if len(findings) < max_findings:
                findings.append(dict(path=json.loads(name), side_a=values[0], side_b=values[1]))
    numeric = None
    if arrays[0] or arrays[1]:
        numeric = compare_arrays(*arrays, metrics=metrics, max_findings=max_findings)
        for finding in numeric['findings']:
            finding['path'] = json.loads(finding['name'])
        coordinate = (numeric.get('metrics') or {}).get('max_coordinate')
        if coordinate is not None:
            coordinate['path'] = json.loads(coordinate['name'])
    return dict(status='completed',
                exact_equal=differences == 0 and (numeric is None or numeric['exact_equal']),
                coverage=dict(inventory='observed_union', side_a_nodes=len(nodes[0]),
                              side_b_nodes=len(nodes[1]), paired_nodes=len(set(nodes[0]) & set(nodes[1])),
                              missing_nodes=dict(side_a=len(names-set(nodes[0])),
                                                 side_b=len(names-set(nodes[1])))),
                metadata=dict(differences=differences, findings=findings,
                              findings_omitted=differences-len(findings)), numeric=numeric)
