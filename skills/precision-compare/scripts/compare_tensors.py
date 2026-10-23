#!/usr/bin/env python3
"""Compare logical observations, gradients or stored updates using artifact schemas 1/2.
Call: compare(C, A, B, mode="metrics"); CLI: python scripts/compare_tensors.py --contract C --side-a A --side-b B --output NEW.json [--mode nonfinite]
Input: Tensor contract and paired manifests: observations, gradients, postclip-gradients or updates.
Output: Global/per-tensor metrics; nonfinite mode returns category counts, sampled locations and finite-subset metrics.
Requires: NumPy; metrics mode checks common-state initial values; nonfinite metrics cover only paired finite coordinates.

Details:
Contract extras: stage=observations|gradients|postclip-gradients|updates,
tensors=[{name,shape,present}] listing logical tensors once.
Observations require present=true and coordinates:{...} on each contract/manifest
tensor. Coordinates identify sample/token selection, boundary, invocation and
execution stage as appropriate; adapters map native records to this identity.
Observation contracts need model/inputs/numerics/scope; state is optional.
Manifest extras: stage, tensors=[{name,shape,present,shards:[...]}].
Shards specify offset (flat logical element index) and value:array_descriptor,
or for updates, before/after:array_descriptors. Offsets cover [0,numel) exactly;
adapters remove native padding and replicas. Absent gradients use present=false
and shards=[]. Stored deltas are FP32/FP64; stored before/after values are
promoted before subtraction.

Updates include events:{model_name:applied|skipped|inactive}. Select active models
in the expected inventory. Before/after pairs allow exact initial-value checks;
delta-only captures use the declared state to identify initial conditions.
A common state requires equal before values; per-side state records permit
task-selected different initial representations and retain their differences.

API: compare(contract,side_a,side_b,chunk_size=1048576,
             *,mode='metrics',max_locations=20).
mode=metrics computes full-tensor metrics and checks update initial values.
mode=nonfinite returns kind=nonfinite_tensors: finite/nan/posinf/neginf counts,
paired-finite metrics and bounded coordinate samples (max_locations per tensor
and globally). For updates it inventories stored values/deltas without checking
initial-value equality.
For already aligned in-memory arrays: stats=Stats(); stats.add(a, b) for each
paired, nonempty, finite, flattened FP64 chunk; stats.report() returns the same
scalar metrics. add() appends coordinates; merge(other) appends another reducer.
max_flat_index refers to that concatenation. Adapters own pairing, coverage,
dtypes and mapping the flat index back to logical coordinates. This API needs no
manifest conversion. Use group_metrics.reduce_groups for named aggregates.

Reports contain global/per-tensor L2 error, RMSE, cosine, maximum error, counts,
norms, events and coordinate findings. An invalid tensor leaves valid per-tensor
results available while its full aggregate is unavailable. Metrics are diagnostics
with thresholds=null. Same-path inputs are labeled self-checks.
CLI: --contract C --side-a A.json --side-b B.json --output NEW.json
     [--mode metrics|nonfinite] [--chunk-size 1048576] [--max-locations 20]
max_locations applies only to nonfinite mode.
Inputs are peer executions identified by manifest side IDs. Reports use schema 2.
Exit: 0=diagnostics, 1=unavailable comparison, 2=CLI/output error.

Nonfinite mode:
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
__all__ = ['compare', 'Stats']

import math
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_io import (Array, ArtifactError, blocked, blocked_status, canonical, cli, contract,
                         integer, manifest, require, result_value, shape_size, unique)


class Stats:
    def __init__(self):
        self.count = 0
        self.scales = [0., 0., 0.]
        self.sums = [0., 0., 0.]
        self.dot = 0.
        self.maximum = 0.
        self.max_index = None

    def add(self, r, c):
        require(r.shape == c.shape and r.size > 0, 'invalid paired chunk')
        require(np.isfinite(r).all() and np.isfinite(c).all(), 'nonfinite tensor observation')
        with np.errstate(over='ignore', invalid='ignore'):
            error = c - r
        require(np.isfinite(error).all(), 'nonfinite tensor difference')
        arrays = [r, c, error]
        old = self.scales[:]
        new = [max(s, float(np.max(np.abs(x)))) for s, x in zip(old, arrays)]
        scaled = [x / s if s else np.zeros_like(x) for x, s in zip(arrays, new)]
        for i in range(3):
            ratio = old[i] / new[i] if new[i] else 0.
            self.sums[i] = math.fsum([self.sums[i] * ratio * ratio, float(np.dot(scaled[i], scaled[i]))])
        factor = (old[0] / new[0]) * (old[1] / new[1]) if new[0] and new[1] else 0.
        self.dot = math.fsum([self.dot * factor, float(np.dot(scaled[0], scaled[1]))])
        index = int(np.argmax(np.abs(error)))
        maximum = abs(float(error[index]))
        if self.max_index is None or maximum > self.maximum:
            self.maximum, self.max_index = maximum, self.count + index
        self.scales = new
        self.count += r.size

    def merge(self, other):
        old = self.scales[:]
        new = [max(a, b) for a, b in zip(old, other.scales)]
        for i, scale in enumerate(new):
            self.sums[i] = math.fsum([self.sums[i] * (old[i] / scale)**2,
                                      other.sums[i] * (other.scales[i] / scale)**2]) if scale else 0.
        if new[0] and new[1]:
            self.dot = math.fsum([self.dot * (old[0]/new[0]) * (old[1]/new[1]),
                                  other.dot * (other.scales[0]/new[0]) * (other.scales[1]/new[1])])
        else:
            self.dot = 0.
        if other.max_index is not None and (self.max_index is None or other.maximum > self.maximum):
            self.maximum, self.max_index = other.maximum, self.count + other.max_index
        self.scales = new
        self.count += other.count

    def report(self):
        if not self.count:
            return {'status': 'not_covered', 'reason': 'no present tensor coordinates', 'count': 0}
        rn, cn, en = [s * math.sqrt(v) for s, v in zip(self.scales, self.sums)]
        rmse = result_value(self.scales[2] * math.sqrt(self.sums[2] / self.count))
        if self.scales[0] and self.scales[1]:
            cosine = self.dot / (math.sqrt(self.sums[0]) * math.sqrt(self.sums[1]))
            require(abs(cosine) <= 1 + 1e-12, 'unstable cosine reduction')
            cosine = result_value(max(-1., min(1., cosine)))
        else:
            cosine = result_value(None, 'zero signal norm')
        return {'count': int(self.count), 'side_a_norm': result_value(rn), 'side_b_norm': result_value(cn),
                'error_norm': result_value(en), 'rmse': rmse, 'cosine': cosine,
                'max_abs': result_value(self.maximum), 'max_flat_index': self.max_index}


class Tensor:
    def __init__(self, base, entry, expected, stage):
        require(entry['shape'] == expected['shape'], 'logical shape mismatch')
        require(type(entry.get('present')) is bool and entry['present'] == expected['present'], 'tensor presence mismatch')
        if stage == 'observations':
            require(canonical(entry.get('coordinates')) == canonical(expected['coordinates']),
                    'observation coordinates differ')
        self.present = entry['present']
        self.size = shape_size(entry['shape'])
        self.shards = []
        require(isinstance(entry.get('shards'), list), 'shards must be a list')
        if not self.present:
            require(stage in ('gradients', 'postclip-gradients') and not entry['shards'], 'absent tensor has values or is an update')
            return
        end = 0
        for shard in sorted(entry['shards'], key=lambda x: x['offset']):
            require(integer(shard['offset']) and shard['offset'] == end, 'shard gap/overlap', 'not_covered')
            if 'value' in shard:
                require('before' not in shard and 'after' not in shard, 'ambiguous shard encoding')
                arrays = [Array(base, shard['value'])]
                require(stage != 'updates' or (not arrays[0].bf16 and arrays[0].dtype.itemsize >= 4), 'stored updates require FP32 or FP64')
            else:
                require(stage == 'updates', 'observations and gradients require value arrays')
                arrays = [Array(base, shard['before']), Array(base, shard['after'])]
                require(arrays[0].shape == arrays[1].shape, 'before/after shape mismatch')
            size = arrays[0].size
            require(size > 0, 'omit empty shards')
            self.shards.append((end, end + size, arrays))
            end += size
        require(end == self.size, 'incomplete tensor coverage', 'not_covered')
        require(self.size > 0, 'empty logical tensor')
        self.cursor = 0

    def chunks(self, start, count):
        """Yield validated stored value or before/after chunks across logical shards."""
        while count:
            while self.shards[self.cursor][1] <= start:
                self.cursor += 1
            lo, hi, arrays = self.shards[self.cursor]
            take = min(count, hi - start)
            yield [array.read(start - lo, take) for array in arrays]
            start += take
            count -= take

    def read(self, start, count):
        parts, initial = [], []
        for arrays in self.chunks(start, count):
            before = arrays[0]
            if len(arrays) == 2:
                after = arrays[1]
                require(np.isfinite(before).all() and np.isfinite(after).all(), 'nonfinite stored parameter')
                parts.append(after-before)
                initial.append(before)
            else:
                parts.append(before)
        return np.concatenate(parts), np.concatenate(initial) if sum(x.size for x in initial) == count else None


def inventory(spec):
    require(spec.get('stage') in ('observations', 'gradients', 'postclip-gradients', 'updates'), 'invalid tensor stage')
    entries = spec.get('tensors')
    require(isinstance(entries, list) and bool(entries), 'empty expected tensor inventory', 'not_covered')
    result = unique(entries, lambda x: x['name'], 'expected tensor')
    for name, item in result.items():
        require(isinstance(name, str) and bool(name), 'invalid tensor name')
        require(shape_size(item['shape']) > 0, 'empty expected tensor')
        require(type(item.get('present')) is bool, 'expected presence must be boolean')
        require(spec['stage'] not in ('updates', 'observations') or item['present'],
                'updates and observations require present tensors')
        if spec['stage'] == 'observations':
            require(isinstance(item.get('coordinates'), dict) and bool(item['coordinates']),
                    'observation coordinates required')
            canonical(item['coordinates'])
    return result


def _update_events(a, b):
    events = [side.get('events') for side in (a, b)]
    for event in events:
        require(isinstance(event, dict) and bool(event), 'missing optimizer events', 'not_covered')
        require(all(v in ('applied', 'skipped', 'inactive') for v in event.values()), 'invalid optimizer event')
    require(set(events[0]) == set(events[1]), 'model event inventory mismatch')
    require(all((events[0][k] == 'inactive') == (events[1][k] == 'inactive') for k in events[0]),
            'active model schedules differ')
    require(any(v != 'inactive' for v in events[0].values()), 'no active optimizer event', 'not_covered')
    return dict(zip(('side_a', 'side_b'), events))


def _compare_metrics(contract_path, side_a_path, side_b_path, chunk_size=1_048_576):
    report = {'kind': 'tensors', 'schema_version': 2, 'status': 'no_thresholds',
              'self_comparison': Path(side_a_path).resolve() == Path(side_b_path).resolve(),
              'thresholds': None, 'per_tensor': {}}
    report['sources'] = {k: str(Path(v).resolve()) for k, v in [('contract', contract_path), ('side_a', side_a_path), ('side_b', side_b_path)]}
    try:
        require(integer(chunk_size, 1), 'chunk_size must be positive')
        spec = contract(contract_path, 'tensors')
        expected = inventory(spec)
        ref, rb = manifest(side_a_path, spec)
        cand, cb = manifest(side_b_path, spec)
        report['sides'] = {'side_a': ref['side'], 'side_b': cand['side']}
        report['conditions'] = {'side_a': ref['_condition_check'], 'side_b': cand['_condition_check']}
        require(report['self_comparison'] or ref['side'] != cand['side'], 'distinct captures require distinct side IDs')
        require(ref.get('stage') == cand.get('stage') == spec['stage'], 'tensor stages differ')
        report['stage'] = spec['stage']
        if spec['stage'] == 'updates':
            report['events'] = _update_events(ref, cand)
        ri = unique(ref['tensors'], lambda x: x['name'], 'tensor')
        ci = unique(cand['tensors'], lambda x: x['name'], 'tensor')
        require(set(ri) == set(ci) == set(expected), 'tensor inventory mismatch', 'not_covered')
        total = Stats()
        failed = False
        max_coordinate = None
        for name in sorted(expected):
            try:
                r = Tensor(rb, ri[name], expected[name], spec['stage'])
                c = r if report['self_comparison'] else Tensor(cb, ci[name], expected[name], spec['stage'])
                if not r.present:
                    report['per_tensor'][name] = {'presence': 'absent_on_both_sides', 'count': 0}
                    continue
                stats = Stats()
                checked_initial = spec['stage'] == 'updates'
                for start in range(0, r.size, chunk_size):
                    count = min(chunk_size, r.size-start)
                    rv, rinitial = r.read(start, count)
                    cv, cinitial = (rv, rinitial) if report['self_comparison'] else c.read(start, count)
                    if rinitial is not None and cinitial is not None:
                        if 'state' in spec['common']:
                            require(np.array_equal(rinitial, cinitial), 'initial stored parameters differ')
                    else:
                        checked_initial = False
                    stats.add(rv, cv)
                result = stats.report()
                if spec['stage'] == 'updates':
                    result['initial_values'] = ('exactly_checked' if 'state' in spec['common'] else 'per_side_state') if checked_initial else 'declared_state_only'
                if spec['stage'] == 'observations':
                    result['coordinates'] = expected[name]['coordinates']
                    result['max_coordinate'] = [int(i) for i in np.unravel_index(stats.max_index, expected[name]['shape'])]
                report['per_tensor'][name] = result
                if total.max_index is None or stats.maximum > total.maximum:
                    if spec['stage'] == 'observations':
                        max_coordinate = {'tensor': name, 'index': stats.max_index,
                                          'coordinate': result['max_coordinate'],
                                          'observation': expected[name]['coordinates']}
                    else:
                        max_coordinate = {'parameter': name, 'index': stats.max_index}
                total.merge(stats)
            except (ArtifactError, OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
                report['per_tensor'][name] = blocked(exc)
                failed = True
        if failed:
            report['status'] = blocked_status(report['per_tensor'].values())
            report['global'] = {'status': 'not_covered', 'reason': 'incomplete valid tensor coverage; no partial aggregate'}
        else:
            report['global'] = total.report()
            report['global']['max_coordinate'] = max_coordinate
            if not total.count:
                report['status'] = 'not_covered'
    except (ArtifactError, OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        report.update(blocked(exc))
    return report


CATEGORIES = ('finite', 'nan', 'posinf', 'neginf')
_DIAGNOSTIC_ERRORS = (ArtifactError, OSError, ValueError, KeyError, TypeError, OverflowError, IndexError)


def _nonfinite_counts():
    return dict.fromkeys(CATEGORIES, 0)


def _nonfinite_categories(values):
    result = np.zeros(values.size, dtype=np.uint8)
    result[np.isnan(values)] = 1
    result[np.isposinf(values)] = 2
    result[np.isneginf(values)] = 3
    return result


def _nonfinite_add_counts(target, codes):
    for key, value in zip(CATEGORIES, np.bincount(codes, minlength=4)):
        target[key] += int(value)


def _nonfinite_location(name, expected, index):
    result = {'tensor': name, 'flat_index': int(index),
              'coordinate': [int(i) for i in np.unravel_index(index, expected['shape'])]}
    if 'coordinates' in expected:
        result['observation'] = expected['coordinates']
    return result


def _nonfinite_read(tensor, start, count):
    values, before_codes, after_codes, generated = [], [], [], 0
    for arrays in tensor.chunks(start, count):
        before = arrays[0]
        if len(arrays) == 2:
            after = arrays[1]
            with np.errstate(over='ignore', invalid='ignore'):
                delta = after - before
            before_codes.append(_nonfinite_categories(before))
            after_codes.append(_nonfinite_categories(after))
            generated += int(np.count_nonzero(np.isfinite(before) & np.isfinite(after)
                                             & ~np.isfinite(delta)))
            values.append(delta)
        else:
            values.append(before)
    return np.concatenate(values), before_codes, after_codes, generated


def _nonfinite_scan(tensor, chunk_size):
    result = {'status': 'no_thresholds', 'presence': 'present' if tensor.present else 'absent',
              'count': 0, 'counts': _nonfinite_counts(), 'stored_before': _nonfinite_counts(),
              'stored_after': _nonfinite_counts(), 'generated_nonfinite_delta': 0,
              'before_after_count': 0}
    if tensor.present:
        tensor.cursor = 0
        for start in range(0, tensor.size, chunk_size):
            values, before, after, generated = _nonfinite_read(tensor, start, min(chunk_size, tensor.size-start))
            _nonfinite_add_counts(result['counts'], _nonfinite_categories(values))
            for codes in before:
                _nonfinite_add_counts(result['stored_before'], codes)
                result['before_after_count'] += int(codes.size)
            for codes in after:
                _nonfinite_add_counts(result['stored_after'], codes)
            result['generated_nonfinite_delta'] += generated
            result['count'] += int(values.size)
    return result


def _nonfinite_unavailable(reason):
    return {'status': 'not_covered', 'reason': reason}


def _nonfinite_paired(a, b, name, expected, chunk_size, max_locations):
    joint = {f'{x}/{y}': 0 for x in CATEGORIES for y in CATEGORIES}
    result = {'status': 'no_thresholds', 'count': 0, 'paired_finite_count': 0,
              'joint_counts': joint, 'samples': []}
    stats, metric_error, maximum_location = Stats(), None, None
    a.cursor = b.cursor = 0
    for start in range(0, a.size, chunk_size):
        size = min(chunk_size, a.size - start)
        av = _nonfinite_read(a, start, size)[0]
        bv = _nonfinite_read(b, start, size)[0]
        ac, bc = _nonfinite_categories(av), _nonfinite_categories(bv)
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
                result['samples'].append({**_nonfinite_location(name, expected, start + int(index)),
                                          'side_a': CATEGORIES[ac[index]], 'side_b': CATEGORIES[bc[index]]})
        if indices.size and metric_error is None:
            try:
                previous_count = stats.count
                previous_max = stats.max_index
                stats.add(av[finite], bv[finite])
                if stats.max_index != previous_max:
                    maximum_location = _nonfinite_location(name, expected, start + int(indices[stats.max_index - previous_count]))
            except _DIAGNOSTIC_ERRORS as exc:
                metric_error = str(exc)
    result['paired_finite_fraction'] = result['paired_finite_count'] / result['count']
    if metric_error is None and stats.count:
        try:
            metrics = stats.report()
            # Stats indexes the compressed finite stream, not the logical tensor.
            metrics.pop('max_flat_index')
            metrics['max_coordinate'] = maximum_location
            result['finite_subset_metrics'] = metrics
        except _DIAGNOSTIC_ERRORS as exc:
            metric_error = str(exc)
    if metric_error or not stats.count:
        result['finite_subset_metrics'] = _nonfinite_unavailable(metric_error or 'no paired finite coordinates')
    return result, stats, metric_error, maximum_location


def _compare_nonfinite(contract_path, side_a_path, side_b_path, chunk_size=1_048_576, max_locations=20):
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
            report['events'] = _update_events(*(item[0] for item in captures))
        inventories = [unique(m['tensors'], lambda x: x['name'], 'tensor') for m, _ in captures]
        unexpected = {key: sorted(set(items) - set(expected)) for key, items in zip(('side_a', 'side_b'), inventories)}
        coverage = {'expected_tensors': len(expected),
                    'expected_present_coordinates': sum(shape_size(e['shape']) for e in expected.values() if e['present']),
                    'diagnosed_tensors': [], 'absent_tensors': [], 'invalid_tensors': [],
                    'unexpected_tensors': unexpected}
        report['coverage'] = coverage
        total, metric_errors, global_max = Stats(), [], None
        aggregate = {'count': 0, 'paired_finite_count': 0, 'side_a': _nonfinite_counts(), 'side_b': _nonfinite_counts(),
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
                    result[key] = _nonfinite_scan(tensor, chunk_size)
                    readers.append(tensor)
                except _DIAGNOSTIC_ERRORS as exc:
                    result[key] = blocked(exc)
            if len(readers) != 2:
                result['status'] = blocked_status((result['side_a'], result['side_b']))
                coverage['invalid_tensors'].append(name)
                continue
            if not item['present']:
                result['status'] = 'no_thresholds'
                result['presence'] = 'absent_on_both_sides'
                result['finite_subset_metrics'] = _nonfinite_unavailable('absent gradients')
                coverage['absent_tensors'].append(name)
                continue
            try:
                pair, stats, error, maximum = _nonfinite_paired(*readers, name, item, chunk_size, max_locations)
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
            except _DIAGNOSTIC_ERRORS as exc:
                result.update(blocked(exc))
                coverage['invalid_tensors'].append(name)
        if coverage['invalid_tensors'] or any(unexpected.values()):
            report['status'] = (blocked_status(report['per_tensor'].values()) if coverage['invalid_tensors'] else 'not_covered')
            report['global'] = _nonfinite_unavailable('incomplete valid tensor coverage; no partial aggregate')
        else:
            aggregate['paired_finite_fraction'] = (aggregate['paired_finite_count'] / aggregate['count']
                                                    if aggregate['count'] else None)
            if metric_errors or not total.count:
                aggregate['finite_subset_metrics'] = _nonfinite_unavailable('; '.join(metric_errors) or 'no paired finite coordinates')
            else:
                try:
                    metrics = total.report()
                    metrics.pop('max_flat_index')
                    metrics['max_coordinate'] = global_max
                    aggregate['finite_subset_metrics'] = metrics
                except _DIAGNOSTIC_ERRORS as exc:
                    aggregate['finite_subset_metrics'] = _nonfinite_unavailable(str(exc))
            report['global'] = aggregate
    except _DIAGNOSTIC_ERRORS as exc:
        report.update(blocked(exc))
    return report


def compare(contract_path, side_a_path, side_b_path, chunk_size=1_048_576,
            *, mode='metrics', max_locations=20):
    """Compare tensor artifacts using full metrics or nonfinite diagnostics."""
    require(mode in ('metrics', 'nonfinite'), 'mode must be metrics or nonfinite')
    if mode == 'nonfinite':
        return _compare_nonfinite(contract_path, side_a_path, side_b_path, chunk_size, max_locations)
    return _compare_metrics(contract_path, side_a_path, side_b_path, chunk_size)


def main(argv=None):
    return cli(compare, __doc__, options={
        'mode': dict(choices=('metrics', 'nonfinite'), default='metrics'),
        'chunk_size': dict(type=int, default=1_048_576),
        'max_locations': dict(type=int, default=20),
    }, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
