#!/usr/bin/env python3
"""Compare task-selected target logprobs; standard library, no dialogue policy.
Call: compare(C, A, B); CLI: python scripts/compare_target_scores.py --contract C --side-a A --side-b B --output NEW.json
Input: Target-score contract and paired records indexed by sample, position and token.
Output: Target-weighted NLL, MAE/P99/max logprob error and coverage.
Requires: Standard library; finite natural logprobs <= 0, exact target inventory and completed rank evidence; EOS allowed.

Details:
Contract: schema_version=1, kind=target-scores, thresholds=null,
common + optional side_conditions provide model/inputs/numerics/scope (see
conditions.py), expected_ranks={side:[rank,...]},
targets=[{sample:str,position:int,token:int}]. Position is a logical target index;
adapters map native masks, labels, actions and cache positions to that index.

Manifest: schema_version=1, kind=target-scores, common, side, expected_ranks,
ranks=[{rank,environment:JSON_path,status:JSON_path}],
scores=[{sample,position,token,logprob}]. Environment records require rank,
versions and actual settings matching the contract; status=completed is required.
Paths resolve relative to the containing manifest. Coordinates must match the
selected targets exactly; EOS and training targets are supported without a
special-token filter. Native losses/reductions remain with the task adapter.

API: compare(contract_path,side_a_path,side_b_path). Reports contain target-weighted
NLL, MAE, linear P99, maximum error/coordinate and complete coverage counts.
CLI: --contract C --side-a A --side-b B --output NEW.json
Exit: 0=metrics computed, 1=unavailable comparison, 2=CLI/output error.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import expected_conditions, execution_settings, load_json, require
from score_metrics import reduce_scores


def coordinate(row):
    require(isinstance(row, dict) and isinstance(row.get('sample'), str) and bool(row['sample']), 'sample ID required')
    require(all(type(row.get(k)) is int and row[k] >= 0 for k in ('position', 'token')), 'invalid target coordinate')
    return row['sample'], row['position'], row['token']


def index(rows):
    require(isinstance(rows, list) and bool(rows), 'target rows required', 'not_covered')
    result = {}
    positions = set()
    for row in rows:
        key = coordinate(row)
        require(key[:2] not in positions, 'duplicate target position')
        positions.add(key[:2])
        result[key] = row
    return result


def compare(contract_path, side_a_path, side_b_path):
    report = dict(kind='target-scores', schema_version=1, status='no_thresholds', thresholds=None,
                  sources={k:str(Path(v).resolve()) for k,v in [('contract',contract_path),('side_a',side_a_path),('side_b',side_b_path)]})
    try:
        spec = load_json(contract_path)
        require(isinstance(spec,dict), 'target-score contract must be an object')
        require(type(spec.get('schema_version')) is int and spec['schema_version'] == 1 and spec.get('kind') == 'target-scores', 'invalid target-score contract')
        require('thresholds' in spec and spec['thresholds'] is None, 'thresholds must be null')
        targets = index(spec.get('targets'))
        sides, values, checks = [], [], []
        for path in (side_a_path, side_b_path):
            data = load_json(path)
            require(isinstance(data,dict), 'target-score manifest must be an object')
            require(type(data.get('schema_version')) is int and data['schema_version'] == 1 and data.get('kind') == 'target-scores', 'invalid target-score manifest')
            side = data.get('side')
            expected_conditions(spec, side, ('model','inputs','numerics','scope'))
            checks.append(execution_settings(spec, data, Path(path).resolve().parent))
            observed = index(data.get('scores'))
            require(set(observed) == set(targets), 'target coordinate coverage mismatch', 'not_covered')
            sides.append(side)
            values.append({key: row['logprob'] for key,row in observed.items()})
        self_check = Path(side_a_path).resolve() == Path(side_b_path).resolve()
        require(self_check or sides[0] != sides[1], 'distinct captures require distinct side IDs')
        metrics = reduce_scores(*values)
        sample, position, token = metrics['max_coordinate']
        metrics['max_coordinate'] = dict(sample=sample, position=position, token=token)
        report.update(sides=dict(side_a=sides[0], side_b=sides[1]), self_comparison=self_check,
                      conditions=dict(side_a=checks[0],side_b=checks[1]), metrics=metrics,
                      coverage=dict(expected=len(targets), paired=len(targets)))
    except FileNotFoundError as exc:
        report.update(status='not_covered', reason=str(exc))
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        report.update(status=getattr(exc,'status','incomparable'), reason=str(exc))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.partition('\n\nDetails:\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('contract','side-a','side-b','output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    try:
        report = compare(args.contract, args.side_a, args.side_b)
        with Path(args.output).open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (ValueError, OSError, TypeError) as exc:
        print(str(exc),file=sys.stderr)
        return 2
    return 0 if report['status']=='no_thresholds' else 1


if __name__=='__main__':
    raise SystemExit(main())
