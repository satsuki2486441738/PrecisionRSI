#!/usr/bin/env python3
"""Compare next-token distributions using artifact schema 1.

Contract extras: vocab_size>=100; normalization_tolerance>0;
renormalization=none|full_vocabulary_fp64;
positions=[{sample:str,position:int,history:[token_id,...]}], with
position=len(history)-1 in canonical causal coordinates.

Manifest positions repeat those coordinates/history and supply a scores array
with row:int. Arrays have shape [positions,vocabulary] or [vocabulary] at row=0;
vocabulary IDs are indices 0..vocab_size-1. Use FP32/FP64 full-vocabulary normalized
natural logprobs before sampling processors. Values are <=0; -inf represents
true zero probability. The selected renormalization policy first validates
original mass, then optionally renormalizes full vocabulary in FP64.

Report: per-position and equal-position aggregate top-100 union/tail metrics,
probability mass and top-1 mismatches. Tail-merged Jensen-Shannon divergence uses natural logs;
complete coordinate sets determine aggregates. Reusing one manifest path labels
the output as a self-check.

CLI: --contract C --side-a REFERENCE.json --side-b CANDIDATE.json --output NEW.json
Inputs are peer executions identified by manifest side IDs. Reports use schema 2.
"""
import math
from pathlib import Path
import numpy as np
from artifact_io import (Array, ArtifactError, aggregate, blocked, blocked_status, cli, contract,
                         integer, manifest, require, result_value, unique)


def coordinate(row):
    require(isinstance(row.get('sample'), str) and row['sample'], 'invalid sample ID')
    require(integer(row.get('position')), 'invalid prediction position')
    return row['sample'], row['position']


def check_history(row, vocab):
    history = row.get('history')
    require(isinstance(history, list) and history and all(integer(x) and x < vocab for x in history), 'invalid history')
    require(len(history) == row['position'] + 1, 'prediction/history offset mismatch')


def lse(values):
    if not values.size:
        return -math.inf
    maximum = float(np.max(values))
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(math.fsum(np.exp(values-maximum)))


def scores(base, record, vocab, tolerance, renormalization="none"):
    source = Array(base, record['scores'])
    require(not source.bf16 and source.dtype.itemsize >= 4, 'logprobs require FP32 or FP64')
    row = record['scores'].get('row', 0)
    require(integer(row), 'invalid score row')
    if len(source.shape) == 1:
        require(source.shape == [vocab] and row == 0, 'vocabulary/row mismatch')
    else:
        require(len(source.shape) == 2 and source.shape[1] == vocab and row < source.shape[0], 'vocabulary/row mismatch')
    values = source.read(row*vocab, vocab)
    require(not np.isnan(values).any() and not np.isposinf(values).any() and (values <= 0).all(), 'invalid logprobs')
    norm = lse(values)
    require(math.isfinite(norm) and abs(math.expm1(norm)) <= tolerance, 'full-vocabulary normalization mismatch')
    require(renormalization in ('none', 'full_vocabulary_fp64'), 'unsupported renormalization')
    return values-norm if renormalization == 'full_vocabulary_fp64' else values, math.exp(norm)


def per_position(lp, lq):
    ids = np.arange(lp.size)
    # Explicit ID tie-breaking, including ties at the 100th cutoff.
    tg = np.lexsort((ids, -lp))[:100]
    tn = np.lexsort((ids, -lq))[:100]
    union = np.union1d(tg, tn)
    tail = np.ones(lp.size, dtype=bool)
    tail[union] = False
    pl = np.append(lp[union], lse(lp[tail]))
    ql = np.append(lq[union], lse(lq[tail]))
    return selected_metrics(union, pl, ql, tg, tn)


def selected_metrics(union, pl, ql, tg, tn, normalization_tolerance=None):
    """Shared arithmetic for validated full-array or offline union-query observations.

    union contains canonical token IDs; pl/ql contain normalized log probabilities
    for each union ID followed by the merged complement. tg/tn retain each side's
    original ordered top-100 IDs. Callers establish full-array provenance first.
    """
    union = np.asarray(union, dtype=np.int64)
    pl, ql = np.asarray(pl, dtype=np.float64), np.asarray(ql, dtype=np.float64)
    tg, tn = np.asarray(tg, dtype=np.int64), np.asarray(tn, dtype=np.int64)
    require(union.ndim == 1 and len(set(union.tolist())) == len(union), 'invalid queried union')
    require(pl.shape == ql.shape == (len(union) + 1,), 'queried score count mismatch')
    require(tg.shape == tn.shape == (100,) and len(set(tg.tolist())) == len(set(tn.tolist())) == 100,
            'queried top-100 inventory mismatch')
    require(set(union.tolist()) == set(tg.tolist()) | set(tn.tolist()), 'queried union coverage mismatch')
    for values in (pl, ql):
        require(not np.isnan(values).any() and not np.isposinf(values).any() and (values <= 0).all(),
                'invalid queried logprobs')
        if normalization_tolerance is not None:
            require(abs(math.expm1(lse(values))) <= normalization_tolerance, 'queried probabilities are not normalized')
    index = {int(token): i for i, token in enumerate(union)}
    for top, values in ((tg, pl), (tn, ql)):
        require(top.tolist() == sorted(top.tolist(), key=lambda token: (-values[index[token]], token)),
                'queried top-100 order mismatch')
        require(all(values[index[int(top[-1])]] >= values[index[int(token)]]
                    for token in union if token not in set(top.tolist())), 'queried top-100 cutoff mismatch')
    mixture = np.logaddexp(pl, ql) - math.log(2)
    terms = []
    for values in (pl, ql):
        active = np.isfinite(values)
        terms.extend((np.exp(values[active]) * (values[active]-mixture[active])).tolist())
    js = result_value(0. if np.array_equal(pl, ql) else max(0., .5*math.fsum(terms)))
    p, q = np.exp(pl[:-1]), np.exp(ql[:-1])
    error = np.abs(q-p)
    worst = int(np.argmax(error))
    top1 = bool(tg[0] == tn[0])
    report = {'union_count': int(union.size), 'tail_merged_js': js, 'top1_agreement': top1,
              'top100_overlap': np.intersect1d(tg, tn).size/100,
              'probability_rmse': math.sqrt(math.fsum(error*error)/union.size),
              'max_probability_error': float(error[worst]), 'max_error_token': int(union[worst]),
              'coverage': {'side_a_top100': math.fsum(p[[index[int(t)] for t in tg]]),
                           'side_b_top100': math.fsum(q[[index[int(t)] for t in tn]]),
                           'side_a_union': math.fsum(p), 'side_b_union': math.fsum(q)}}
    if not top1:
        tokens = sorted([int(tg[0]), int(tn[0])])
        report['top1_mismatch'] = {'side_a_token': int(tg[0]), 'side_b_token': int(tn[0]),
                                 'probabilities': [{'token': i, 'side_a': float(p[index[i]]),
                                                   'side_b': float(q[index[i]])} for i in tokens],
                                 'side_a_margin': float(p[index[int(tg[0])]]-p[index[int(tg[1])]]),
                                 'side_b_margin': float(q[index[int(tn[0])]]-q[index[int(tn[1])]])}
    return report


def summarize(rows):
    result = {}
    for field in ('probability_rmse', 'max_probability_error'):
        values = [r[field] for r in rows]
        result[field] = aggregate(values)
        maximum = rows[max(range(len(rows)), key=lambda i: values[i])]
        result[field]['max_coordinate'] = {k: maximum[k] for k in ('sample', 'position')}
        if field == 'max_probability_error':
            result[field]['max_coordinate']['token'] = maximum['max_error_token']
    js = [r['tail_merged_js']['value'] for r in rows]
    result['tail_merged_js'] = aggregate(js)
    worst = rows[max(range(len(rows)), key=lambda i: js[i])]
    result['tail_merged_js']['max_coordinate'] = {k: worst[k] for k in ('sample', 'position')}
    matches = sum(r['top1_agreement'] for r in rows)
    result['top1_agreement'] = {'count': matches, 'total': len(rows), 'fraction': matches/len(rows)}
    result['top100_overlap'] = {'mean': math.fsum(r['top100_overlap'] for r in rows)/len(rows)}
    result['coverage'] = {}
    for key in rows[0]['coverage']:
        values = [r['coverage'][key] for r in rows]
        result['coverage'][key] = {'mean': math.fsum(values)/len(values), 'p1': float(np.quantile(values, .01, method='linear')), 'min': min(values)}
    return result


def compare(contract_path, side_a_path, side_b_path):
    report = {'kind': 'distributions', 'schema_version': 2, 'status': 'no_thresholds', 'thresholds': None,
              'self_comparison': Path(side_a_path).resolve() == Path(side_b_path).resolve(), 'positions': []}
    report['sources'] = {k: str(Path(v).resolve()) for k, v in [('contract', contract_path), ('side_a', side_a_path), ('side_b', side_b_path)]}
    try:
        spec = contract(contract_path, 'distributions')
        vocab, tolerance = spec['vocab_size'], spec['normalization_tolerance']
        renormalization = spec['renormalization']
        require(renormalization in ('none', 'full_vocabulary_fp64'), 'unsupported renormalization')
        require(integer(vocab, 100), 'vocabulary must contain at least 100 IDs')
        require(type(tolerance) in (int, float) and math.isfinite(tolerance) and 0 < tolerance < 1, 'invalid normalization tolerance')
        require(isinstance(spec.get('positions'), list) and spec['positions'], 'empty expected positions', 'not_covered')
        expected = unique(spec['positions'], coordinate, 'expected position')
        for row in expected.values():
            check_history(row, vocab)
        ref, rb = manifest(side_a_path, spec)
        cand, cb = manifest(side_b_path, spec)
        report['sides'] = {'side_a': ref['side'], 'side_b': cand['side']}
        require(report['self_comparison'] or ref['side'] != cand['side'], 'distinct captures require distinct side IDs')
        ri, ci = [unique(side['positions'], coordinate, 'position') for side in (ref, cand)]
        require(set(ri) == set(ci) == set(expected), 'position coverage mismatch', 'not_covered')
        failed = False
        for key in sorted(expected):
            row = {'sample': key[0], 'position': key[1]}
            try:
                for entry in (ri[key], ci[key]):
                    check_history(entry, vocab)
                    require(entry['history'] == expected[key]['history'], 'history mismatch')
                lp, pn = scores(rb, ri[key], vocab, tolerance, renormalization)
                lq, qn = (lp, pn) if report['self_comparison'] else scores(cb, ci[key], vocab, tolerance, renormalization)
                row.update(per_position(lp, lq))
                row['normalization_mass'] = {'side_a': pn, 'side_b': qn}
            except (ArtifactError, OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
                row.update(blocked(exc))
                failed = True
            report['positions'].append(row)
        report['position_count'] = len(expected)
        if failed:
            report['status'] = blocked_status(report['positions'])
            report['aggregate'] = {'status': 'not_covered', 'reason': 'incomplete valid position coverage; no partial aggregate'}
        else:
            report['aggregate'] = summarize(report['positions'])
        report['normalization_tolerance'] = tolerance
        report['renormalization'] = renormalization
    except (ArtifactError, OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        report.update(blocked(exc))
    return report


if __name__ == '__main__':
    raise SystemExit(cli(compare, __doc__))
