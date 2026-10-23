"""Reduce validated coordinate observations globally and by task-defined groups.
Call: reduce_groups(expected, groups, reducer, *observations)
Input: Independent coordinates, group-to-coordinate maps, a reducer and observation maps.
Output: global and groups reports with independent coverage/metrics.
Requires: Standard library; coordinates must belong to expected; groups may overlap; incomplete groups have no aggregate.

Details:
reduce_groups(expected, groups, reducer, *observations) uses only the standard
library. expected is a nonempty iterable of unique nonempty names (strings) or
coordinate tuples containing strings/integers. groups maps group names to iterables;
groups may overlap or be empty. Each observation is a coordinate-to-value mapping
whose keys belong to expected. Adapters own coordinate meanings, group membership
and validation of the values (including execution conditions and query digests).

The read-only reducer receives one lazy coordinate-to-value Mapping per observation;
values are read only when the reducer accesses them, including NPZ array values.
Use array_compare.compare_arrays directly for named arrays, or
score_metrics.reduce_scores for paired target scores; for validated
per-position distribution metrics use a reducer such as
lambda rows: compare_distributions.summarize(list(rows.values())). The original
observations determine each aggregate, including percentiles and worst positions.

Returns {global: result, groups: {name: result}}. Each result has status,
coverage={expected_count, observed_counts, paired_count, missing}, and metrics.
A complete nonempty group is completed; empty/incomplete groups are not_covered
with metrics=null. Complete groups remain available when another group is
incomplete. Missing coordinates are listed per observation in canonical JSON
coordinate order. Duplicate/unknown coordinates or invalid definitions raise
ValueError; reducer errors propagate. Inputs are not modified by this tool.
"""
from collections.abc import Mapping
import json


def _order(key):
    return json.dumps(key, ensure_ascii=False, separators=(',', ':'))


def _coordinates(values):
    if isinstance(values, (str, bytes)):
        raise ValueError('coordinates must be an iterable of names or tuples')
    result = set()
    for key in values:
        if not ((isinstance(key, str) and key) or
                (isinstance(key, tuple) and key and
                 all(type(part) in (str, int) for part in key))):
            raise ValueError('coordinates must be nonempty names or tuples of strings/integers')
        if key in result:
            raise ValueError('duplicate coordinate')
        result.add(key)
    return result


class _Subset(Mapping):
    def __init__(self, source, keys):
        self._source = source
        self._keys = tuple(keys)
        self._selected = frozenset(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._selected

    def __getitem__(self, key):
        if key not in self._selected:
            raise KeyError(key)
        return self._source[key]


def reduce_groups(expected, groups, reducer, *observations):
    """Apply one existing reducer to original observations at selected coordinates."""
    expected = _coordinates(expected)
    if not expected or not observations:
        raise ValueError('nonempty expected coordinates and observations required')
    if not isinstance(groups, Mapping) or not callable(reducer):
        raise ValueError('group mapping and callable reducer required')
    sources = []
    for values in observations:
        if not isinstance(values, Mapping):
            raise ValueError('observations must map coordinates to values')
        keys = _coordinates(values)
        if keys - expected:
            raise ValueError('observation coordinates outside expected set')
        sources.append(keys)
    memberships = {}
    for name, values in groups.items():
        if not isinstance(name, str) or not name:
            raise ValueError('group names must be nonempty strings')
        keys = _coordinates(values)
        if keys - expected:
            raise ValueError('group coordinates outside expected set')
        memberships[name] = keys

    def reduce(keys):
        ordered = sorted(keys, key=_order)
        missing = [sorted(keys-source, key=_order) for source in sources]
        coverage = dict(expected_count=len(keys),
                        observed_counts=[len(keys & source) for source in sources],
                        paired_count=len(keys.intersection(*sources)),
                        missing=[[key if isinstance(key, str) else list(key) for key in rows] for rows in missing])
        if not keys or any(missing):
            return dict(status='not_covered', coverage=coverage, metrics=None)
        metrics = reducer(*[_Subset(values, ordered) for values in observations])
        return dict(status='completed', coverage=coverage, metrics=metrics)

    return {
        'global': reduce(expected),
        'groups': {name: reduce(keys) for name, keys in memberships.items()}}
