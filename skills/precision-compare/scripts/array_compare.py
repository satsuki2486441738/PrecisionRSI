#!/usr/bin/env python3
"""Compare already corresponding named arrays without artifact conversion (NumPy).

API: compare_arrays(side_a, side_b, expected=None, metrics=False,
                    max_findings=8, chunk_size=1048576) -> JSON-ready report.
Sides map unique logical names to real numeric arrays/scalars. Adapters establish
observation semantics, inputs/state, masks and logical correspondence. Native
loaders can supply mappings directly; names are paired literally. expected is an
optional independent inventory of names; otherwise the observed union is used,
which cannot detect omissions shared by both sides. Inputs are not modified.

exact_equal requires complete inventory, equal shapes/dtypes, finite values and
numeric equality in native dtype (signed zeros compare equal, not byte identity).
Coverage, unequal-element/nonfinite counts and bounded findings are always
returned. metrics=True additionally uses the existing FP64 Stats reducer for L2,
RMSE, cosine and maximum-error coordinates. Incomplete/incompatible/nonfinite
inputs have no full aggregate. Integers outside [-2**53,2**53] retain exact
comparison but no FP64 metrics. Arrays are visited in sorted-name, C-flat order
using bounded chunks, including noncontiguous arrays. NPZ loading materializes
one named array at a time; use native mappings/memmaps for larger observations.

CLI: array_compare.py A.npz B.npz --output NEW.json [--metrics]
                     [--expected names.json] [--max-findings 8]
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


def compare_arrays(side_a, side_b, expected=None, metrics=False, max_findings=8,
                   chunk_size=1_048_576):
    """Pair named native arrays; compute exact equality and optional FP64 metrics."""
    if (type(metrics) is not bool or type(max_findings) is not int or max_findings < 0
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
    missing = [names - k for k in keys]
    extra = [k - names for k in keys]
    covered = not any(missing + extra)
    coverage = dict(expected_arrays=len(names), side_a_arrays=len(keys[0]),
                    side_b_arrays=len(keys[1]), paired_arrays=len(names & keys[0] & keys[1]),
                    missing={s: len(k) for s, k in zip(('side_a', 'side_b'), missing)},
                    unexpected={s: len(k) for s, k in zip(('side_a', 'side_b'), extra)},
                    compared_arrays=0, compared_elements=0)
    findings, finding_count = [], 0
    def finding(name, reason, **details):
        nonlocal finding_count
        finding_count += 1
        if len(findings) < max_findings:
            findings.append(dict(name=name, reason=reason, **details))
    for side, absent, unexpected in zip(('side_a', 'side_b'), missing, extra):
        for name in sorted(absent):
            finding(name, 'missing', side=side)
        for name in sorted(unexpected):
            finding(name, 'unexpected', side=side)
    stats = Stats() if metrics else None
    metric_reason = None if covered else 'incomplete inventory'
    incompatible = unequal = unequal_arrays = 0
    nonfinite = dict(side_a=0, side_b=0)
    segments = []
    for name in sorted(names & keys[0] & keys[1]):
        a, b = (np.asarray(side[name]) for side in (side_a, side_b))
        if a.dtype.kind not in 'biuf' or b.dtype.kind not in 'biuf':
            raise ValueError('real numeric arrays required: ' + name)
        if a.shape != b.shape or a.dtype != b.dtype:
            incompatible += 1
            finding(name, 'shape_or_dtype', side_a_shape=list(a.shape), side_b_shape=list(b.shape),
                    side_a_dtype=str(a.dtype), side_b_dtype=str(b.dtype))
            metric_reason = 'incompatible arrays'
            continue
        coverage['compared_arrays'] += 1
        coverage['compared_elements'] += a.size
        segments.append((name, a.shape, a.size))
        first_bad = first_unequal = None
        for start in range(0, a.size, chunk_size):
            x, y = a.flat[start:start + chunk_size], b.flat[start:start + chunk_size]
            finite = [np.isfinite(v) for v in (x, y)]
            for side, valid in zip(nonfinite, finite):
                nonfinite[side] += int(valid.size - np.count_nonzero(valid))
            valid = finite[0] & finite[1]
            if not valid.all():
                metric_reason = 'nonfinite arrays'
                if first_bad is None:
                    first_bad = start + int(np.argmin(valid))
            different = x != y
            unequal += int(np.count_nonzero(different))
            if different.any() and first_unequal is None:
                first_unequal = start + int(np.argmax(different))
            if metrics and metric_reason is None:
                if x.dtype.kind in 'iu' and any(int(v.min()) < -2**53 or int(v.max()) > 2**53 for v in (x, y)):
                    metric_reason = 'integer magnitude exceeds exact FP64 range'
                else:
                    try:
                        stats.add(x.astype(np.float64), y.astype(np.float64))
                    except ValueError as exc:
                        metric_reason = str(exc)
        unequal_arrays += first_unequal is not None
        for reason, index in (('nonfinite', first_bad), ('values', first_unequal)):
            if index is not None:
                finding(name, reason, index=[int(i) for i in np.unravel_index(index, a.shape)])
    complete = covered and incompatible == 0
    report = dict(status='completed' if complete else ('incomparable' if incompatible else 'not_covered'),
                  exact_equal=complete and unequal == 0 and not any(nonfinite.values()),
                  coverage=coverage, unequal_arrays=unequal_arrays, unequal_elements=unequal,
                  incompatible_arrays=incompatible, nonfinite=nonfinite,
                  findings=findings, findings_omitted=finding_count - len(findings))
    if metrics:
        values = stats.report() if metric_reason is None else None
        if values and values.get('max_flat_index') is not None:
            index = values['max_flat_index']
            for name, shape, size in segments:
                if index < size:
                    values['max_coordinate'] = dict(name=name, index=[int(i) for i in np.unravel_index(index, shape)])
                    break
                index -= size
        report.update(metrics=values, metrics_unavailable=metric_reason)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('side_a')
    parser.add_argument('side_b')
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected')
    parser.add_argument('--metrics', action='store_true')
    parser.add_argument('--max-findings', type=int, default=8)
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
