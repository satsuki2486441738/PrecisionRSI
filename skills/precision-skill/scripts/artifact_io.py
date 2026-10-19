"""Artifact schema 1 and bounded-memory NumPy/safetensors readers.

Contract fields: schema_version=1|2, kind=tensors|distributions, thresholds=null,
common plus optional side_conditions (see conditions.py) provide model, inputs,
numerics, scope and state for tensors;
expected_ranks={side_id:[int,...],...}. Side IDs are nonempty strings naming
execution setups; hardware/runtime details belong in environment records.
Common entries contain identities or settings; rank sets come from the contract.

Manifest fields: schema_version, kind, common, side=side_id,
expected_ranks=[int,...], ranks=[{rank,environment:JSON_path,status:JSON_path}].
Environment documents contain rank and a nonempty versions object; status
records contain rank and status=completed. Schema 2 requires environment.actual
settings matching common + side_conditions; schema 1 checks them when present.
Resolve paths relative to each JSON.

Array descriptors: {path:array.npy} or {path:array.safetensors,key:tensor_name}.
Readers accept contiguous F16/BF16/F32/F64 arrays and flat logical shard offsets.
JSON uses unique keys and finite numeric literals. NumPy is the runtime dependency.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np
from conditions import expected_conditions, execution_settings, known


class ArtifactError(ValueError):
    def __init__(self, message, status='incomparable'):
        super().__init__(message)
        self.status = status


def require(condition, message, status='incomparable'):
    if not condition:
        raise ArtifactError(message, status)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def unique(items, key, label):
    result = {}
    for item in items:
        k = key(item)
        require(k not in result, f'duplicate {label}: {k}')
        result[k] = item
    return result


def load_json(path):
    def pairs(values):
        out = {}
        for key, value in values:
            require(key not in out, f'duplicate JSON key: {key}')
            out[key] = value
        return out
    def constant(value):
        raise ArtifactError(f'nonfinite JSON value: {value}')
    try:
        with Path(path).open() as stream:
            return json.load(stream, object_pairs_hook=pairs, parse_constant=constant,
                             parse_float=lambda s: float(s) if math.isfinite(float(s)) else constant(s))
    except FileNotFoundError as exc:
        raise ArtifactError(f'missing file: {path}', 'not_covered') from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f'invalid JSON: {path}: {exc}') from exc


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def contract(path, kind):
    data = load_json(path)
    require(isinstance(data, dict), 'expected JSON object')
    require(type(data.get('schema_version')) is int and data['schema_version'] in (1, 2), 'schema_version must be 1 or 2')
    require(data.get('kind') == kind, f'expected kind {kind}')
    require('thresholds' in data and data['thresholds'] is None, 'only null thresholds supported')
    common = data.get('common')
    require(isinstance(common, dict), 'common must be an object')
    keys = ['model', 'inputs', 'numerics', 'scope'] + (['state'] if kind == 'tensors' else [])
    ranks = data.get('expected_ranks')
    require(isinstance(ranks, dict) and ranks and all(isinstance(side, str) and side.strip() for side in ranks), 'contract must declare expected ranks keyed by nonempty side IDs', 'not_covered')
    for values in ranks.values():
        require(isinstance(values, list) and values and all(integer(r) for r in values) and len(values) == len(set(values)), 'invalid contract ranks')
    for side in ranks:
        expected_conditions(data, side, keys)
    return data


def manifest(path, spec, expected_side=None):
    path = Path(path).resolve()
    data = load_json(path)
    require(isinstance(data, dict), 'expected JSON object')
    require(type(data.get('schema_version')) is int and data['schema_version'] == spec['schema_version'], 'manifest schema_version mismatch')
    require(data.get('kind') == spec['kind'], 'manifest kind mismatch')
    require(canonical(data.get('common')) == canonical(spec['common']), 'common conditions differ')
    require(isinstance(data.get('side'), str) and data['side'].strip() and data['side'] in spec['expected_ranks'], 'side must name an execution declared in contract expected_ranks')
    if expected_side:
        require(data['side'] == expected_side, f'expected {expected_side} manifest')
    data['_condition_check'] = execution_settings(spec, data, path.parent, load_json,
        require_actual=spec['schema_version'] == 2 or bool(spec.get('side_conditions')))
    return data, path.parent


def shape_size(shape):
    require(isinstance(shape, list) and all(integer(n) for n in shape), 'invalid shape')
    return math.prod(shape)


class Array:
    """Read-only metadata plus chunk reads; no whole-tensor allocation."""
    def __init__(self, base, descriptor):
        self.path = (Path(base) / descriptor['path']).resolve()
        self.bf16 = False
        self.offset = 0
        require(self.path.is_file(), f'missing array: {self.path}', 'not_covered')
        if self.path.suffix == '.npy':
            array = np.load(self.path, mmap_mode='r', allow_pickle=False)
            if not isinstance(array, np.memmap):
                if hasattr(array, 'close'):
                    array.close()
                raise ArtifactError('expected a memory-mappable .npy array')
            try:
                require(array.dtype.kind == 'f' and array.dtype.itemsize in (2, 4, 8), 'unsupported array dtype')
                require(array.flags.c_contiguous, 'array must be C contiguous')
                self.shape = list(array.shape)
                self.dtype = array.dtype
                self.offset = array.offset
            finally:
                array._mmap.close()
        elif self.path.suffix == '.safetensors':
            with self.path.open('rb') as stream:
                prefix = stream.read(8)
                require(len(prefix) == 8, 'truncated safetensors header')
                length = struct.unpack('<Q', prefix)[0]
                require(0 < length <= 100_000_000 and 8 + length <= self.path.stat().st_size, 'invalid safetensors header length')
                header = json.loads(stream.read(length))
            key = descriptor.get('key')
            require(key in header and key != '__metadata__', 'missing safetensors key', 'not_covered')
            entry = header[key]
            dtype = entry['dtype']
            require(dtype in ('F16', 'BF16', 'F32', 'F64'), 'unsupported safetensors dtype')
            self.bf16 = dtype == 'BF16'
            self.dtype = np.dtype({'F16': '<f2', 'BF16': '<u2', 'F32': '<f4', 'F64': '<f8'}[dtype])
            self.shape = entry['shape']
            count = shape_size(self.shape)
            start, end = entry['data_offsets']
            require(integer(start) and integer(end) and end - start == count * self.dtype.itemsize, 'invalid tensor offsets')
            self.offset = 8 + length + start
            require(8 + length + end <= self.path.stat().st_size, 'truncated safetensors data')
        else:
            raise ArtifactError('only .npy and .safetensors arrays supported')
        self.size = shape_size(self.shape)
        require(self.offset + self.size * self.dtype.itemsize <= self.path.stat().st_size, 'truncated array')

    def read(self, start, count):
        require(integer(start) and integer(count) and start + count <= self.size, 'array slice out of bounds')
        if count == 0:
            return np.empty(0, dtype=np.float64)
        mapped = np.memmap(self.path, mode='r', dtype=self.dtype, offset=self.offset + start * self.dtype.itemsize, shape=(count,))
        try:
            if self.bf16:
                return (mapped.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            return mapped.astype(np.float64)
        finally:
            mapped._mmap.close()


def blocked(exc):
    return {'status': getattr(exc, 'status', 'incomparable'), 'reason': str(exc)}


def result_value(value, reason=None):
    if value is not None and not math.isfinite(value):
        value, reason = None, reason or 'nonfinite metric'
    return {'value': value, 'status': 'incomparable' if reason else 'no_thresholds', **({'reason': reason} if reason else {})}


def blocked_status(results):
    statuses = {r.get('status') for r in results}
    return next((s for s in ('execution_failed', 'incomparable', 'not_covered') if s in statuses), 'incomparable')


def aggregate(values):
    require(bool(values), 'empty metric coverage', 'not_covered')
    return {'mean': math.fsum(values) / len(values), 'p99': float(np.quantile(values, .99, method='linear')), 'max': max(values)}


def cli(compare, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--side-a', required=True)
    parser.add_argument('--side-b', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        report = compare(args.contract, args.side_a, args.side_b)
        with Path(args.output).open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if report['status'] == 'no_thresholds' else 1
