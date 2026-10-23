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
"""
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

    def read(self, start, count):
        parts, initial = [], []
        remaining = count
        while remaining:
            while self.shards[self.cursor][1] <= start:
                self.cursor += 1
            lo, hi, arrays = self.shards[self.cursor]
            take = min(remaining, hi - start)
            before = arrays[0].read(start-lo, take)
            if len(arrays) == 2:
                after = arrays[1].read(start-lo, take)
                require(np.isfinite(before).all() and np.isfinite(after).all(), 'nonfinite stored parameter')
                parts.append(after-before)
                initial.append(before)
            else:
                parts.append(before)
            start += take
            remaining -= take
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
            for side in (ref, cand):
                require(isinstance(side.get('events'), dict) and side['events'], 'missing optimizer events', 'not_covered')
                require(all(v in ('applied', 'skipped', 'inactive') for v in side['events'].values()), 'invalid optimizer event')
            require(set(ref['events']) == set(cand['events']), 'model event inventory mismatch')
            require(all((ref['events'][k] == 'inactive') == (cand['events'][k] == 'inactive') for k in ref['events']), 'active model schedules differ')
            require(any(v != 'inactive' for v in ref['events'].values()), 'no active optimizer event', 'not_covered')
            report['events'] = {'side_a': ref['events'], 'side_b': cand['events']}
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


def compare(contract_path, side_a_path, side_b_path, chunk_size=1_048_576,
            *, mode='metrics', max_locations=20):
    """Compare tensor artifacts using full metrics or nonfinite diagnostics."""
    require(mode in ('metrics', 'nonfinite'), 'mode must be metrics or nonfinite')
    if mode == 'nonfinite':
        from inspect_nonfinite import compare as diagnose
        return diagnose(contract_path, side_a_path, side_b_path, chunk_size, max_locations)
    return _compare_metrics(contract_path, side_a_path, side_b_path, chunk_size)


def main(argv=None):
    return cli(compare, __doc__, options={
        'mode': dict(choices=('metrics', 'nonfinite'), default='metrics'),
        'chunk_size': dict(type=int, default=1_048_576),
        'max_locations': dict(type=int, default=20),
    }, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
