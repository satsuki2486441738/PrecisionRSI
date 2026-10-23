#!/usr/bin/env python3
"""Compare already corresponding named arrays without artifact conversion (NumPy).
Call: compare_arrays(a, b, expected=None, metrics=False); CLI: python scripts/array_compare.py A.npz B.npz --output NEW.json [--metrics]
Input: Corresponding name-to-numeric-array maps, or NPZ files.
Output: exact_equal, coverage, bounded findings; optional FP64 error metrics. --all-findings returns every differing/nonfinite coordinate.
Requires: NumPy; equality requires matching shapes/dtypes and finite values. expected detects shared omissions.

Details:
API: compare_arrays(side_a, side_b, expected=None, metrics=False,
                    max_findings=8, chunk_size=1048576) -> JSON-ready report.
Sides map unique logical names to real numeric arrays/scalars. Adapters establish
observation semantics, inputs/state, masks and logical correspondence. Native
loaders can supply mappings directly; names are paired literally. expected is an
optional independent inventory of names; otherwise the observed union is used,
which cannot detect omissions shared by both sides. Inputs are not modified.

Report fields: status, exact_equal, coverage, unequal_arrays, unequal_elements,
incompatible_arrays, dtype_differences, nonfinite, findings, findings_omitted. coverage contains
expected_arrays, side_a_arrays, side_b_arrays, paired_arrays, missing/unexpected
counts by side, compared_arrays and compared_elements. Findings have name, reason
and applicable index/values or shape/dtype details. With metrics=True, metrics
and metrics_unavailable are added; metrics holds norms/errors, max_coordinate
(name/index), and max_values. Undefined norms retain the existing null/reason rules.

exact_equal requires complete inventory, equal shapes/dtypes, finite values and
numeric equality (signed zeros compare equal, not byte identity). Different dtypes
are recorded and prevent exact_equal, but same-shape real arrays still receive
value comparisons and optional metrics. Cross-dtype integer comparisons preserve
exact values before any metric conversion.
Coverage, unequal-element/nonfinite counts and bounded findings are always
returned. Numeric findings include values={dtype, side_a, side_b}; values are
exact scalar repr strings, including nonfinite values and large integers. dtype
is a string for matching dtypes, otherwise {side_a: dtype, side_b: dtype}.
metrics=True additionally uses the existing FP64 Stats reducer for L2,
RMSE, cosine, maximum-error coordinates and max_values in the same representation. Incomplete/shape-incompatible/nonfinite
inputs have no full aggregate. Integers outside [-2**53,2**53] retain exact
comparison but no FP64 metrics. Arrays are visited in sorted-name, C-flat order
using bounded chunks, including noncontiguous arrays. max_findings=None returns
every differing or nonfinite element once, with its native values; the default
retains bounded representative findings. Exhaustive output memory scales with
the number of findings. NPZ loading materializes
one named array at a time; use native mappings/memmaps for larger observations.

CLI: array_compare.py A.npz B.npz --output NEW.json [--metrics]
                     [--expected names.json] [--max-findings 8 | --all-findings]
The CLI reads NPZ files with allow_pickle=False; no task directory layout is
assumed. Native Torch/BF16 state needs an adapter preserving its dtype identity.
Exit 0=complete comparison (including unequal values), 1=incomplete/incompatible,
2=invalid input/output. Inspect exact_equal for equality, not the exit code.
"""
import argparse
from contextlib import ExitStack
import json
from collections.abc import Mapping
from pathlib import Path
import sys

import numpy as np
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_tensors import Stats
from conditions import load_json


def _names(values):
    names = list(values)
    if any(not isinstance(k, str) or not k for k in names) or len(set(names)) != len(names):
        raise ValueError('logical names must be unique nonempty strings')
    return set(names)


def _values(a, b, index):
    dtype = str(a.dtype) if a.dtype == b.dtype else dict(side_a=str(a.dtype), side_b=str(b.dtype))
    return dict(dtype=dtype, side_a=repr(a.flat[index].item()),
                side_b=repr(b.flat[index].item()))


def _different(a, b):
    if a.dtype != b.dtype and (a.dtype.kind in 'iu' or b.dtype.kind in 'iu'):
        # NumPy may promote signed/unsigned integers or integer/float pairs to
        # FP64, hiding exact differences above 2**53. Python scalar equality does not.
        return np.fromiter((x.item() != y.item() for x, y in zip(a, b)),
                           dtype=bool, count=a.size)
    return a != b


class _Comparison:
    """Merge per-array reductions without rereading values or retaining arrays."""
    def __init__(self, names, keys, metrics, max_findings):
        missing = [names - k for k in keys]
        extra = [k - names for k in keys]
        self.covered = not any(missing + extra)
        self.coverage = dict(expected_arrays=len(names), side_a_arrays=len(keys[0]),
                             side_b_arrays=len(keys[1]), paired_arrays=len(names & keys[0] & keys[1]),
                             missing={s: len(k) for s, k in zip(('side_a', 'side_b'), missing)},
                             unexpected={s: len(k) for s, k in zip(('side_a', 'side_b'), extra)},
                             compared_arrays=0, compared_elements=0)
        self.findings, self.finding_count, self.limit = [], 0, max_findings
        for side, absent, unexpected in zip(('side_a', 'side_b'), missing, extra):
            for name in sorted(absent):
                self.finding(name, 'missing', side=side)
            for name in sorted(unexpected):
                self.finding(name, 'unexpected', side=side)
        self.stats = Stats() if metrics else None
        self.metric_reason = None if self.covered else 'incomplete inventory'
        self.incompatible = self.dtype_differences = self.unequal = self.unequal_arrays = 0
        self.nonfinite = dict(side_a=0, side_b=0)
        self.max_coordinate = self.max_values = None

    def finding(self, name, reason, **details):
        self.finding_count += 1
        if self.limit is None or len(self.findings) < self.limit:
            self.findings.append(dict(name=name, reason=reason, **details))

    def merge(self, other):
        for field in ('compared_arrays', 'compared_elements'):
            self.coverage[field] += other.coverage[field]
        for field in ('incompatible', 'dtype_differences', 'unequal', 'unequal_arrays'):
            setattr(self, field, getattr(self, field) + getattr(other, field))
        for side in self.nonfinite:
            self.nonfinite[side] += other.nonfinite[side]
        self.findings.extend(other.findings if self.limit is None else
                             other.findings[:max(0, self.limit - len(self.findings))])
        self.finding_count += other.finding_count
        if other.metric_reason:
            self.metric_reason = other.metric_reason
        if self.stats is not None:
            if other.stats.max_index is not None and (self.stats.max_index is None or
                                                       other.stats.maximum > self.stats.maximum):
                self.max_coordinate, self.max_values = other.max_coordinate, other.max_values
            self.stats.merge(other.stats)

    def report(self):
        complete = self.covered and self.incompatible == 0
        result = dict(status='completed' if complete else ('incomparable' if self.incompatible else 'not_covered'),
                      exact_equal=complete and self.dtype_differences == 0 and self.unequal == 0 and not any(self.nonfinite.values()),
                      coverage=self.coverage, unequal_arrays=self.unequal_arrays, unequal_elements=self.unequal,
                      incompatible_arrays=self.incompatible, dtype_differences=self.dtype_differences,
                      nonfinite=self.nonfinite, findings=self.findings,
                      findings_omitted=self.finding_count - len(self.findings))
        if self.stats is not None:
            values = self.stats.report() if self.metric_reason is None else None
            if values and self.max_coordinate is not None:
                values.update(max_coordinate=dict(self.max_coordinate), max_values=dict(self.max_values))
            result.update(metrics=values, metrics_unavailable=self.metric_reason)
        return result


def _scan_array(name, a, b, metrics, max_findings, chunk_size):
    result = _Comparison({name}, ({name}, {name}), metrics, max_findings)
    if a.dtype.kind not in 'biuf' or b.dtype.kind not in 'biuf':
        raise ValueError('real numeric arrays required: ' + name)
    if a.shape != b.shape:
        result.incompatible = 1
        result.finding(name, 'shape_or_dtype', side_a_shape=list(a.shape), side_b_shape=list(b.shape),
                       side_a_dtype=str(a.dtype), side_b_dtype=str(b.dtype))
        result.metric_reason = 'incompatible arrays'
        return result
    if a.dtype != b.dtype:
        result.dtype_differences = 1
        result.finding(name, 'dtype', side_a_dtype=str(a.dtype), side_b_dtype=str(b.dtype))
    result.coverage.update(compared_arrays=1, compared_elements=a.size)
    first_bad = first_unequal = None
    for start in range(0, a.size, chunk_size):
        x, y = a.flat[start:start + chunk_size], b.flat[start:start + chunk_size]
        finite = [np.isfinite(v) for v in (x, y)]
        for side, valid in zip(result.nonfinite, finite):
            result.nonfinite[side] += int(valid.size - np.count_nonzero(valid))
        valid = finite[0] & finite[1]
        if not valid.all():
            result.metric_reason = 'nonfinite arrays'
            if first_bad is None:
                first_bad = start + int(np.argmin(valid))
        different = _different(x, y)
        result.unequal += int(np.count_nonzero(different))
        if max_findings is None:
            for offset in np.flatnonzero(different | ~valid):
                index = start + int(offset)
                result.finding(name, 'values' if valid[offset] else 'nonfinite',
                               index=[int(i) for i in np.unravel_index(index, a.shape)],
                               values=_values(a, b, index))
        if different.any() and first_unequal is None:
            first_unequal = start + int(np.argmax(different))
        if metrics and result.metric_reason is None:
            if any(v.dtype.kind in 'iu' and (int(v.min()) < -2**53 or int(v.max()) > 2**53) for v in (x, y)):
                result.metric_reason = 'integer magnitude exceeds exact FP64 range'
            else:
                try:
                    result.stats.add(x.astype(np.float64), y.astype(np.float64))
                except ValueError as exc:
                    result.metric_reason = str(exc)
    result.unequal_arrays = int(first_unequal is not None)
    for reason, index in (('nonfinite', first_bad), ('values', first_unequal)):
        if max_findings is not None and index is not None:
            result.finding(name, reason, index=[int(i) for i in np.unravel_index(index, a.shape)],
                           values=_values(a, b, index))
    if metrics and result.stats.max_index is not None:
        index = result.stats.max_index
        result.max_coordinate = dict(name=name, index=[int(i) for i in np.unravel_index(index, a.shape)])
        result.max_values = _values(a, b, index)
    return result


def _compare_partitions(side_a, side_b, expected=None, metrics=False, max_findings=8,
                        chunk_size=1_048_576, partitions=None):
    if (type(metrics) is not bool
            or (max_findings is not None and (type(max_findings) is not int or max_findings < 0))
            or type(chunk_size) is not int or chunk_size < 1):
        raise ValueError('invalid metrics, max_findings or chunk_size')
    if not all(isinstance(side, Mapping) for side in (side_a, side_b)):
        raise ValueError('sides must map logical names to arrays')
    keys = [_names(side) for side in (side_a, side_b)]
    if isinstance(expected, (str, bytes)):
        raise ValueError('expected must be an inventory of names')
    names = keys[0] | keys[1] if expected is None else _names(expected)
    if not names:
        raise ValueError('nonempty expected or observed inventory required')
    total = _Comparison(names, keys, metrics, max_findings)
    groups, owners = {}, {}
    for group, selected in (partitions or {}).items():
        selected = set(selected)
        groups[group] = _Comparison(selected, [k & selected for k in keys], metrics, max_findings)
        for name in selected:
            if name not in names or name in owners:
                raise ValueError('partitions must contain disjoint expected names')
            owners[name] = group
    for name in sorted(names & keys[0] & keys[1]):
        a, b = (np.asarray(side[name]) for side in (side_a, side_b))
        part = _scan_array(name, a, b, metrics, max_findings, chunk_size)
        total.merge(part)
        if name in owners:
            groups[owners[name]].merge(part)
    return total.report(), {key: group.report() for key, group in groups.items()}


def compare_arrays(side_a, side_b, expected=None, metrics=False, max_findings=8,
                   chunk_size=1_048_576):
    """Pair named native arrays; compute exact equality and optional FP64 metrics."""
    return _compare_partitions(side_a, side_b, expected, metrics, max_findings, chunk_size)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.partition('\n\nDetails:\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('side_a')
    parser.add_argument('side_b')
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected')
    parser.add_argument('--metrics', action='store_true')
    findings = parser.add_mutually_exclusive_group()
    findings.add_argument('--max-findings', type=int, default=8)
    findings.add_argument('--all-findings', dest='max_findings', action='store_const', const=None,
                          help='return every differing/nonfinite coordinate and native values')
    args = parser.parse_args()
    try:
        with ExitStack() as stack:
            sides = [stack.enter_context(np.load(p, allow_pickle=False)) for p in (args.side_a, args.side_b)]
            expected = load_json(args.expected) if args.expected else None
            result = compare_arrays(*sides, expected=expected, metrics=args.metrics, max_findings=args.max_findings)
        with Path(args.output).open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return 0 if result['status'] == 'completed' else 1
    except (ValueError, TypeError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
