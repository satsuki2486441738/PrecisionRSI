#!/usr/bin/env python3
"""Stream complete logical tensors. Requires NumPy; see artifact_io for shared schema.

Contract extras: stage=gradients|updates, tensors=[{name, shape, present}]. The
independent expected inventory must contain every logical parameter once.
Manifest extras: stage, tensors=[{name, shape, present, shards:[...]}].
Each nonempty shard has offset (flat logical element index) and either
value:array_descriptor, or (updates only) before/after:array_descriptors.
Offsets must cover [0,numel) exactly, without padding/replicas; layouts may differ
between sides. Missing gradients use present=false, shards=[], never zero fill.
Stored deltas require FP32/FP64; before/after values are promoted before subtraction.
Updates also require events:{model_name: applied|skipped|inactive}; events are
reported separately from finite numerical errors. Inactive models should not be
in the expected inventory. Exact initial values are checked when both sides
provide before/after; delta-only inputs rely on reviewed common.state evidence.

compare(contract, reference, candidate, chunk_size=1048576) returns global and
per-tensor relative L2/cosine/max, counts and norms, no acceptance thresholds.
Any invalid tensor blocks global metrics, retaining independent valid tensors.
Same manifest path is a labeled self-check, not a cross-device result.
CLI: --contract C --reference GPU --candidate NPU --output NEW.json
Exit 0=diagnostics, 1=blocked scope, 2=CLI/output error.
"""
import math
from pathlib import Path
import numpy as np
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
        if self.scales[0]:
            # Log-domain ratio avoids overflow in scale ratio for finite final results.
            log_ratio = math.log(self.scales[2]) - math.log(self.scales[0]) + .5*(math.log(self.sums[2])-math.log(self.sums[0])) if self.scales[2] else -math.inf
            rel = math.exp(log_ratio) if log_ratio < 709.782712893384 else math.inf
            relative = result_value(rel)
        else:
            relative = result_value(None, 'zero reference norm with nonzero error') if self.scales[2] else result_value(0.)
        if self.scales[0] and self.scales[1]:
            cosine = self.dot / math.sqrt(self.sums[0]) / math.sqrt(self.sums[1])
            require(abs(cosine) <= 1 + 1e-12, 'unstable cosine reduction')
            cosine = result_value(max(-1., min(1., cosine)))
        else:
            cosine = result_value(None, 'zero signal norm')
        return {'count': int(self.count), 'reference_norm': result_value(rn), 'candidate_norm': result_value(cn),
                'error_norm': result_value(en), 'relative_l2': relative, 'cosine': cosine,
                'max_abs': result_value(self.maximum), 'max_flat_index': self.max_index}


class Tensor:
    def __init__(self, base, entry, expected, stage):
        require(entry['shape'] == expected['shape'], 'logical shape mismatch')
        require(type(entry.get('present')) is bool and entry['present'] == expected['present'], 'gradient presence mismatch')
        self.present = entry['present']
        self.size = shape_size(entry['shape'])
        self.shards = []
        require(isinstance(entry.get('shards'), list), 'shards must be a list')
        if not self.present:
            require(stage == 'gradients' and not entry['shards'], 'absent tensor has values or is an update')
            return
        end = 0
        for shard in sorted(entry['shards'], key=lambda x: x['offset']):
            require(integer(shard['offset']) and shard['offset'] == end, 'shard gap/overlap', 'not_covered')
            if 'value' in shard:
                require('before' not in shard and 'after' not in shard, 'ambiguous shard encoding')
                arrays = [Array(base, shard['value'])]
                require(stage != 'updates' or (not arrays[0].bf16 and arrays[0].dtype.itemsize >= 4), 'stored updates require FP32 or FP64')
            else:
                require(stage == 'updates', 'gradients require value arrays')
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
    require(spec.get('stage') in ('gradients', 'updates'), 'invalid tensor stage')
    entries = spec.get('tensors')
    require(isinstance(entries, list) and bool(entries), 'empty expected tensor inventory', 'not_covered')
    result = unique(entries, lambda x: x['name'], 'expected parameter')
    for name, item in result.items():
        require(isinstance(name, str) and bool(name), 'invalid parameter name')
        require(shape_size(item['shape']) > 0, 'empty expected tensor')
        require(type(item.get('present')) is bool, 'expected presence must be boolean')
        require(spec['stage'] == 'gradients' or item['present'], 'updates must cover all parameters')
    return result


def compare(contract_path, reference_path, candidate_path, chunk_size=1_048_576):
    report = {'kind': 'tensors', 'schema_version': 1, 'status': 'no_thresholds',
              'self_comparison': Path(reference_path).resolve() == Path(candidate_path).resolve(),
              'thresholds': None, 'per_tensor': {}}
    report['sources'] = {k: str(Path(v).resolve()) for k, v in [('contract', contract_path), ('reference', reference_path), ('candidate', candidate_path)]}
    try:
        require(integer(chunk_size, 1), 'chunk_size must be positive')
        spec = contract(contract_path, 'tensors')
        expected = inventory(spec)
        ref, rb = manifest(reference_path, spec, 'GPU')
        cand, cb = manifest(candidate_path, spec, 'GPU' if report['self_comparison'] else 'NPU')
        require(ref.get('stage') == cand.get('stage') == spec['stage'], 'tensor stages differ')
        report['stage'] = spec['stage']
        if spec['stage'] == 'updates':
            for side in (ref, cand):
                require(isinstance(side.get('events'), dict) and side['events'], 'missing optimizer events', 'not_covered')
                require(all(v in ('applied', 'skipped', 'inactive') for v in side['events'].values()), 'invalid optimizer event')
            require(set(ref['events']) == set(cand['events']), 'model event inventory mismatch')
            require(all((ref['events'][k] == 'inactive') == (cand['events'][k] == 'inactive') for k in ref['events']), 'active model schedules differ')
            require(any(v != 'inactive' for v in ref['events'].values()), 'no active optimizer event', 'not_covered')
            report['events'] = {'reference': ref['events'], 'candidate': cand['events']}
        ri = unique(ref['tensors'], lambda x: x['name'], 'parameter')
        ci = unique(cand['tensors'], lambda x: x['name'], 'parameter')
        require(set(ri) == set(ci) == set(expected), 'parameter inventory mismatch', 'not_covered')
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
                        require(np.array_equal(rinitial, cinitial), 'initial stored parameters differ')
                    else:
                        checked_initial = False
                    stats.add(rv, cv)
                result = stats.report()
                if spec['stage'] == 'updates':
                    result['initial_values'] = 'exactly_checked' if checked_initial else 'declared_state_only'
                report['per_tensor'][name] = result
                if total.max_index is None or stats.maximum > total.maximum:
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


if __name__ == '__main__':
    raise SystemExit(cli(compare, __doc__))
