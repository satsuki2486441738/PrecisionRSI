#!/usr/bin/env python3
"""Nonfinite tensor diagnostics, also exposed by compare_tensors --mode nonfinite.

API: compare(contract_path, side_a_path, side_b_path, chunk_size=1048576,
             max_locations=20). CLI: --contract --side-a --side-b --output,
with optional --chunk-size and --max-locations. Output is strict finite JSON;
exit 0 means diagnostics collected, 1 means invalid/incomplete artifacts or
parameters, and 2 means CLI/output failure. Output paths must be new.

Supports observations, gradients, postclip-gradients and updates, schema 1/2,
NumPy/safetensors (including BF16), independent logical shard partitions and
absent gradients via the existing Tensor/Array validation. Reads are streaming,
O(chunk_size) working memory, with two passes for present tensors. Categories
are finite/nan/posinf/neginf; joint_counts includes all 16 ordered pairs.
Samples select the first nonfinite logical coordinates in tensor-name/flat-index
order, capped by max_locations per tensor AND in the global sample list (0
suppresses samples). Observation identity is preserved alongside C-order indices.
Counts and sample selection are independent of chunk size; floating reductions
have the usual floating-point rounding dependence.

Updates accept stored deltas or before/after arrays, even mixed across shards.
Before/after subtraction is promoted to float64. stored_before/stored_after
count invalid stored values; generated_nonfinite_delta counts only coordinates
with BOTH stored inputs finite whose delta becomes nonfinite. Delta categories
include results from invalid stored inputs too, and do not imply a valid update.
No initial-parameter equality or optimizer correctness is asserted; declared
state/conditions and optimizer events are retained, with event validation.

Finite-subset Stats metrics use only coordinates finite on both sides, and their
maximum is mapped back to the original logical coordinate. No paired finite
values, finite subtraction overflow or unstable reductions make metrics
unavailable with a reason, without losing category counts. Nonfinite metric
scalars use Stats' explicit unavailable value/reason. Invalid/missing tensors
retain independent valid-side diagnostics but block full global counts/metrics;
coverage lists every expected tensor and unexpected manifest entry. Partial
counts are never presented as full coverage. No thresholds or full-tensor
correctness verdicts are produced; no framework repair is performed.
"""
from pathlib import Path

import numpy as np

from artifact_io import (ArtifactError, blocked, blocked_status, cli, contract, integer,
                         manifest, require, shape_size, unique)
from compare_tensors import Stats, Tensor, inventory

CATEGORIES = ('finite', 'nan', 'posinf', 'neginf')
ERRORS = (ArtifactError, OSError, ValueError, KeyError, TypeError, OverflowError, IndexError)


def counts():
    return dict.fromkeys(CATEGORIES, 0)


def categories(values):
    result = np.zeros(values.size, dtype=np.uint8)
    result[np.isnan(values)] = 1
    result[np.isposinf(values)] = 2
    result[np.isneginf(values)] = 3
    return result


def add_counts(target, codes):
    for key, value in zip(CATEGORIES, np.bincount(codes, minlength=4)):
        target[key] += int(value)


def location(name, expected, index):
    result = {'tensor': name, 'flat_index': int(index),
              'coordinate': [int(i) for i in np.unravel_index(index, expected['shape'])]}
    if 'coordinates' in expected:
        result['observation'] = expected['coordinates']
    return result


def read(tensor, start, count):
    """Use Tensor-validated shard metadata, retaining raw update provenance."""
    values, before_codes, after_codes, generated = [], [], [], 0
    while count:
        while tensor.shards[tensor.cursor][1] <= start:
            tensor.cursor += 1
        lo, hi, arrays = tensor.shards[tensor.cursor]
        take = min(count, hi - start)
        before = arrays[0].read(start - lo, take)
        if len(arrays) == 2:
            after = arrays[1].read(start - lo, take)
            with np.errstate(over='ignore', invalid='ignore'):
                delta = after - before
            before_codes.append(categories(before))
            after_codes.append(categories(after))
            generated += int(np.count_nonzero(np.isfinite(before) & np.isfinite(after)
                                             & ~np.isfinite(delta)))
            values.append(delta)
        else:
            values.append(before)
        start += take
        count -= take
    return (np.concatenate(values), before_codes, after_codes, generated)


def scan(tensor, chunk_size):
    result = {'status': 'no_thresholds', 'presence': 'present' if tensor.present else 'absent',
              'count': 0, 'counts': counts(), 'stored_before': counts(),
              'stored_after': counts(), 'generated_nonfinite_delta': 0,
              'before_after_count': 0}
    if tensor.present:
        tensor.cursor = 0
        for start in range(0, tensor.size, chunk_size):
            values, before, after, generated = read(tensor, start, min(chunk_size, tensor.size-start))
            add_counts(result['counts'], categories(values))
            for codes in before:
                add_counts(result['stored_before'], codes)
                result['before_after_count'] += int(codes.size)
            for codes in after:
                add_counts(result['stored_after'], codes)
            result['generated_nonfinite_delta'] += generated
            result['count'] += int(values.size)
    return result


def unavailable(reason):
    return {'status': 'not_covered', 'reason': reason}


def paired(a, b, name, expected, chunk_size, max_locations):
    joint = {f'{x}/{y}': 0 for x in CATEGORIES for y in CATEGORIES}
    result = {'status': 'no_thresholds', 'count': 0, 'paired_finite_count': 0,
              'joint_counts': joint, 'samples': []}
    stats, metric_error, maximum_location = Stats(), None, None
    a.cursor = b.cursor = 0
    for start in range(0, a.size, chunk_size):
        size = min(chunk_size, a.size - start)
        av = read(a, start, size)[0]
        bv = read(b, start, size)[0]
        ac, bc = categories(av), categories(bv)
        bins = np.bincount(ac * 4 + bc, minlength=16)
        for key, value in zip(joint, bins):
            joint[key] += int(value)
        finite = (ac == 0) & (bc == 0)
        indices = np.flatnonzero(finite)
        result['count'] += size
        result['paired_finite_count'] += int(indices.size)
        remaining = max_locations - len(result['samples'])
        if remaining:
            for index in np.flatnonzero(~finite)[:remaining]:
                result['samples'].append({**location(name, expected, start + int(index)),
                                          'side_a': CATEGORIES[ac[index]], 'side_b': CATEGORIES[bc[index]]})
        if indices.size and metric_error is None:
            try:
                previous_count = stats.count
                previous_max = stats.max_index
                stats.add(av[finite], bv[finite])
                if stats.max_index != previous_max:
                    maximum_location = location(name, expected, start + int(indices[stats.max_index - previous_count]))
            except ERRORS as exc:
                metric_error = str(exc)
    result['paired_finite_fraction'] = result['paired_finite_count'] / result['count']
    if metric_error is None and stats.count:
        try:
            metrics = stats.report()
            # Stats indexes the compressed finite stream, not the logical tensor.
            metrics.pop('max_flat_index')
            metrics['max_coordinate'] = maximum_location
            result['finite_subset_metrics'] = metrics
        except ERRORS as exc:
            metric_error = str(exc)
    if metric_error or not stats.count:
        result['finite_subset_metrics'] = unavailable(metric_error or 'no paired finite coordinates')
    return result, stats, metric_error, maximum_location


def compare(contract_path, side_a_path, side_b_path, chunk_size=1_048_576, max_locations=20):
    report = {'kind': 'nonfinite_tensors', 'schema_version': 2, 'status': 'no_thresholds',
              'thresholds': None, 'per_tensor': {}}
    try:
        require(integer(chunk_size, 1), 'chunk_size must be a positive integer')
        require(integer(max_locations), 'max_locations must be a nonnegative integer')
        paths = (contract_path, side_a_path, side_b_path)
        report['sources'] = {key: str(Path(path).resolve()) for key, path in
                             zip(('contract', 'side_a', 'side_b'), paths)}
        report['self_comparison'] = Path(side_a_path).resolve() == Path(side_b_path).resolve()
        spec = contract(contract_path, 'tensors')
        expected = inventory(spec)
        captures = [manifest(path, spec) for path in paths[1:]]
        report['sides'] = {key: capture[0]['side'] for key, capture in zip(('side_a', 'side_b'), captures)}
        report['conditions'] = {key: capture[0]['_condition_check'] for key, capture in zip(('side_a', 'side_b'), captures)}
        require(report['self_comparison'] or report['sides']['side_a'] != report['sides']['side_b'],
                'distinct captures require distinct side IDs')
        require(all(m.get('stage') == spec['stage'] for m, _ in captures), 'tensor stages differ')
        report['stage'] = spec['stage']
        if spec['stage'] == 'updates':
            events = [m.get('events') for m, _ in captures]
            for event in events:
                require(isinstance(event, dict) and bool(event), 'missing optimizer events', 'not_covered')
                require(all(v in ('applied', 'skipped', 'inactive') for v in event.values()), 'invalid optimizer event')
            require(set(events[0]) == set(events[1]), 'model event inventory mismatch')
            require(all((events[0][k] == 'inactive') == (events[1][k] == 'inactive') for k in events[0]),
                    'active model schedules differ')
            require(any(v != 'inactive' for v in events[0].values()), 'no active optimizer event', 'not_covered')
            report['events'] = dict(zip(('side_a', 'side_b'), events))
        inventories = [unique(m['tensors'], lambda x: x['name'], 'tensor') for m, _ in captures]
        unexpected = {key: sorted(set(items) - set(expected)) for key, items in zip(('side_a', 'side_b'), inventories)}
        coverage = {'expected_tensors': len(expected),
                    'expected_present_coordinates': sum(shape_size(e['shape']) for e in expected.values() if e['present']),
                    'diagnosed_tensors': [], 'absent_tensors': [], 'invalid_tensors': [],
                    'unexpected_tensors': unexpected}
        report['coverage'] = coverage
        total, metric_errors, global_max = Stats(), [], None
        aggregate = {'count': 0, 'paired_finite_count': 0, 'side_a': counts(), 'side_b': counts(),
                     'joint_counts': {f'{x}/{y}': 0 for x in CATEGORIES for y in CATEGORIES}, 'samples': []}
        for name, item in sorted(expected.items()):
            result, readers = {}, []
            report['per_tensor'][name] = result
            if 'coordinates' in item:
                result['coordinates'] = item['coordinates']
            for key, items, (_, base) in zip(('side_a', 'side_b'), inventories, captures):
                try:
                    require(name in items, f'missing tensor: {name}', 'not_covered')
                    tensor = Tensor(base, items[name], item, spec['stage'])
                    result[key] = scan(tensor, chunk_size)
                    readers.append(tensor)
                except ERRORS as exc:
                    result[key] = blocked(exc)
            if len(readers) != 2:
                result['status'] = blocked_status((result['side_a'], result['side_b']))
                coverage['invalid_tensors'].append(name)
                continue
            if not item['present']:
                result['status'] = 'no_thresholds'
                result['presence'] = 'absent_on_both_sides'
                result['finite_subset_metrics'] = unavailable('absent gradients')
                coverage['absent_tensors'].append(name)
                continue
            try:
                pair, stats, error, maximum = paired(*readers, name, item, chunk_size, max_locations)
                result.update(pair)
                coverage['diagnosed_tensors'].append(name)
                for key in ('count', 'paired_finite_count'):
                    aggregate[key] += pair[key]
                for key in ('side_a', 'side_b'):
                    for category in CATEGORIES:
                        aggregate[key][category] += result[key]['counts'][category]
                for key in aggregate['joint_counts']:
                    aggregate['joint_counts'][key] += pair['joint_counts'][key]
                aggregate['samples'].extend(pair['samples'][:max_locations-len(aggregate['samples'])])
                if error:
                    metric_errors.append(f'{name}: {error}')
                elif stats.count:
                    if total.max_index is None or stats.maximum > total.maximum:
                        global_max = maximum
                    total.merge(stats)
            except ERRORS as exc:
                result.update(blocked(exc))
                coverage['invalid_tensors'].append(name)
        if coverage['invalid_tensors'] or any(unexpected.values()):
            report['status'] = (blocked_status(report['per_tensor'].values()) if coverage['invalid_tensors'] else 'not_covered')
            report['global'] = unavailable('incomplete valid tensor coverage; no partial aggregate')
        else:
            aggregate['paired_finite_fraction'] = (aggregate['paired_finite_count'] / aggregate['count']
                                                    if aggregate['count'] else None)
            if metric_errors or not total.count:
                aggregate['finite_subset_metrics'] = unavailable('; '.join(metric_errors) or 'no paired finite coordinates')
            else:
                try:
                    metrics = total.report()
                    metrics.pop('max_flat_index')
                    metrics['max_coordinate'] = global_max
                    aggregate['finite_subset_metrics'] = metrics
                except ERRORS as exc:
                    aggregate['finite_subset_metrics'] = unavailable(str(exc))
            report['global'] = aggregate
    except ERRORS as exc:
        report.update(blocked(exc))
    return report


def main(argv=None):
    return cli(compare, __doc__, options={
        'chunk_size': dict(type=int, default=1_048_576),
        'max_locations': dict(type=int, default=20),
    }, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
