#!/usr/bin/env python3
"""Replay the three anonymous recorded target comparisons with the stdlib.

Usage: python replay-targets.py selected-targets.json
Reproduces target metrics only. Archived distribution summaries are not replayed.
The 1e-12 verification tolerance is not a model-acceptance threshold.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys


def require(ok, message):
    if not ok:
        raise ValueError(message)


def metrics(a, b):
    errors = [abs(x-y) for x, y in zip(a, b)]
    ordered = sorted(errors)
    h = .99 * (len(errors)-1)
    low, high = math.floor(h), math.ceil(h)
    na, nb = math.fsum(-v for v in a)/len(a), math.fsum(-v for v in b)/len(b)
    return {'target_count': len(errors),
            'side_a_nll_nat_per_token': na, 'side_b_nll_nat_per_token': nb,
            'nll_abs_diff_nat_per_token': abs(na-nb),
            'mae_nat': math.fsum(errors)/len(errors),
            'p99_nat': ordered[low]+(h-low)*(ordered[high]-ordered[low]),
            'max_abs_nat': max(errors),
            'max_anonymous_row_index_zero_based': errors.index(max(errors))}


def replay(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    require(data.get('schema_version') == 1 and
            data.get('kind') == 'anonymous_npu_repeat_target_replay', 'unsupported evidence schema')
    require(data.get('thresholds') is None and data.get('acceptance') is None and
            data.get('full_core_complete') is False, 'unexpected acceptance or completion claim')
    require(data.get('h100_torch29_comparison') == 'pending', 'this bundle has no GPU/NPU result')
    w = data['workload']
    expected = {'parameter_dtype':'BF16', 'attention':'eager', 'seed':0,
                'parallelism':'TP1', 'independent_inputs':1, 'prompt_tokens_including_bos':128,
                'fixed_continuation_tokens':32, 'fresh_process_repeats':3,
                'prefill_calls_per_capture':1, 'single_token_decode_calls_per_capture':31,
                'target_positions_per_capture':32, 'native_logits_retained':False}
    require(all(k in w and type(w[k]) is type(v) and w[k] == v for k, v in expected.items()),
            'workload scope differs')
    require(data['local_replay']['full_vocabulary_rows_checked'] == 0 and
            data['local_replay']['full_vocabulary_rows_transferred'] == 0 and
            data['archived_remote_distributions']['remote_full_vocabulary_rows_checked'] == 96,
            'local versus archived distribution scope differs')
    runs = data['repeats']
    require(isinstance(runs, list) and len(runs) == 3 and
            [r['id'] for r in runs] == ['npu-r1','npu-r2','npu-r3'], 'three ordered repeats required')
    indexed = {}
    for run in runs:
        values = run['target_logprobs_nat']
        require(isinstance(values, list) and len(values) == 32 and
                all(type(v) in (int, float) and math.isfinite(v) and v <= 0 for v in values),
                '32 finite nonpositive anonymous target logprobs required')
        indexed[run['id']] = values
    pairs = data['target_pairs']
    expected_pairs = [list(p) for p in itertools.combinations(indexed, 2)]
    require(isinstance(pairs, list) and [p['sides'] for p in pairs] == expected_pairs,
            'all three distinct ordered repeat pairs required')
    checked = []
    for pair in pairs:
        require(pair['coverage'] == {'expected':32, 'paired':32}, 'target coverage differs')
        a, b = pair['sides']
        result = metrics(indexed[a], indexed[b])
        recorded = pair['metrics']
        require(set(result) == set(recorded), 'recorded metric inventory differs')
        for key, actual in result.items():
            reference = recorded[key]
            require(type(reference) in (int, float) and math.isfinite(reference), 'invalid reference: '+key)
            if type(actual) is int:
                require(type(reference) is int and reference == actual, 'integer metric differs: '+key)
            else:
                require(math.isclose(actual, reference, rel_tol=0, abs_tol=1e-12), 'metric differs: '+key)
        checked.append({'sides':pair['sides'], 'coverage':pair['coverage'], 'metrics':result})
    return {'status':'recorded_target_metrics_replayed',
            'evidence_sha256':hashlib.sha256(raw).hexdigest(), 'pairs':checked,
            'absolute_verification_tolerance':1e-12,
            'local_full_vocabulary_rows_checked':0,
            'archived_distribution_metrics_replayed':False,
            'thresholds':None, 'acceptance':None, 'full_core_complete':False}


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: python replay-targets.py selected-targets.json')
    print(json.dumps(replay(sys.argv[1]), indent=2, allow_nan=False))
