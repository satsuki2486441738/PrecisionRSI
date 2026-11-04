#!/usr/bin/env python3
"""Replay one paired target-logprob comparison using only the standard library.

Usage: python replay-target-logprobs.py target-logprob-evidence.json
The file contains derived scores, not text, token IDs, raw logits or distributions.
Successful checks reproduce the recorded metrics, not numerical acceptance.
"""
import hashlib
import json
import math
import pathlib
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: python replay-target-logprobs.py target-logprob-evidence.json')
    payload = pathlib.Path(sys.argv[1]).read_bytes()
    data = json.loads(payload)
    if data['schema_version'] != 1 or data['thresholds'] is not None or data['acceptance'] is not None:
        raise ValueError('unsupported schema or unexpected acceptance')
    rows = data['pairs']
    if data['coverage'] != {'expected': 32, 'paired': 32} or len(rows) != 32:
        raise ValueError('expected 32 paired target scores')
    if [r['pair_index_zero_based'] for r in rows] != list(range(32)):
        raise ValueError('pair indices must be unique and ordered')
    a = [r['h100_logprob_nat'] for r in rows]
    b = [r['rtx5090_logprob_nat'] for r in rows]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v > 0 for v in a+b):
        raise ValueError('target log probabilities must be finite and non-positive')
    errors = [abs(x-y) for x,y in zip(a,b)]
    ordered = sorted(errors)
    n = len(errors)
    h = 0.99 * (n-1)
    low, high = math.floor(h), math.ceil(h)
    result = {
        'mae_nat': math.fsum(errors)/n,
        'p99_nat': ordered[low] + (h-low)*(ordered[high]-ordered[low]),
        'max_abs_nat': max(errors),
        'h100_nll_nat_per_token': math.fsum(-v for v in a)/n,
        'rtx5090_nll_nat_per_token': math.fsum(-v for v in b)/n,
        'max_pair_index_zero_based': errors.index(max(errors)),
    }
    recorded = data['expected_metrics']
    if set(result) != set(recorded):
        raise ValueError('unexpected reference metrics')
    for name, actual in result.items():
        reference = recorded[name]
        if not isinstance(reference, (int, float)) or not math.isfinite(reference):
            raise ValueError('nonfinite reference: '+name)
        if name == 'max_pair_index_zero_based':
            if type(reference) is not int or actual != reference:
                raise ValueError('maximum index mismatch')
        elif not math.isclose(actual, reference, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('metric mismatch: '+name)
    print(json.dumps({'status': 'verified', 'input_sha256': hashlib.sha256(payload).hexdigest(),
                      'paired': n, 'metrics': result, 'absolute_tolerance': 1e-12,
                      'thresholds': None, 'acceptance': None}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()