#!/usr/bin/env python3
"""Validate one execution side's artifact inventory and values.

Uses the tensor/distribution schemas in artifact_io.py, compare_tensors.py and
compare_distributions.py. Reads each required array and checks coordinates,
ranks, shapes, dtypes and numerical values. Returns status=validated for complete
valid inputs. Requires NumPy.

CLI: --contract C --manifest M --output NEW.json
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from artifact_io import (ArtifactError, blocked, contract, integer, load_json,
                         manifest, require, unique)
from compare_tensors import Tensor, inventory
from compare_distributions import coordinate, check_history, scores


def validate(contract_path, manifest_path, chunk_size=1_048_576):
    report = {'status': 'validated', 'evidence_scope': 'single-side artifact checks only', 'items': {}}
    try:
        require(integer(chunk_size, 1), 'chunk_size must be positive')
        kind = load_json(contract_path)['kind']
        require(kind in ('tensors', 'distributions'), 'unsupported artifact kind')
        spec = contract(contract_path, kind)
        data, base = manifest(manifest_path, spec)
        report.update(kind=kind, side=data['side'], conditions=data['_condition_check'])
        if kind == 'tensors':
            expected = inventory(spec)
            require(data.get('stage') == spec['stage'], 'tensor stage mismatch')
            if spec['stage'] == 'updates':
                require(isinstance(data.get('events'), dict) and data['events'] and all(v in ('applied', 'skipped', 'inactive') for v in data['events'].values()), 'invalid/missing optimizer events')
                require(any(v != 'inactive' for v in data['events'].values()), 'no active optimizer event', 'not_covered')
            observed = unique(data['tensors'], lambda x: x['name'], 'tensor')
            require(set(expected) == set(observed), 'tensor inventory mismatch', 'not_covered')
            for name, entry in expected.items():
                tensor = Tensor(base, observed[name], entry, spec['stage'])
                if tensor.present:
                    for start in range(0, tensor.size, chunk_size):
                        values, _ = tensor.read(start, min(chunk_size, tensor.size-start))
                        require(np.isfinite(values).all(), f'nonfinite tensor: {name}')
                report['items'][name] = {'count': tensor.size if tensor.present else 0, 'present': tensor.present}
        else:
            vocab, tolerance = spec['vocab_size'], spec['normalization_tolerance']
            require(integer(vocab, 100), 'vocabulary must contain at least 100 IDs')
            require(type(tolerance) in (int, float) and np.isfinite(tolerance) and 0 < tolerance < 1, 'invalid normalization tolerance')
            expected = unique(spec['positions'], coordinate, 'expected position')
            require(bool(expected), 'empty position coverage', 'not_covered')
            observed = unique(data['positions'], coordinate, 'position')
            require(set(expected) == set(observed), 'position coverage mismatch', 'not_covered')
            for key, row in expected.items():
                check_history(row, vocab)
                check_history(observed[key], vocab)
                require(observed[key]['history'] == row['history'], 'history mismatch')
                _, mass = scores(base, observed[key], vocab, tolerance, spec['renormalization'])
                report['items'][json.dumps(key)] = {'count': vocab, 'normalization_mass': mass}
        report['item_count'] = len(report['items'])
        report['coordinate_count'] = sum(item['count'] for item in report['items'].values())
    except (ArtifactError, OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        report.update(blocked(exc))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        report = validate(args.contract, args.manifest)
        with Path(args.output).open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if report['status'] == 'validated' else 1


if __name__ == '__main__':
    raise SystemExit(main())
