"""Shared execution-condition checks; standard library.

Contracts contain common:{field:value}, expected_ranks:{side_id:[rank,...]}, and
optional side_conditions:{side_id:{field:value}}. A field belongs either to
common or to each side's conditions, never both. Nested objects are field values.
Actual settings come from each rank's environment.actual object; extra observed
fields are retained by the collector. Required fields may be common or per-side.
check_actual verifies expected values without treating declared differences as
errors. JSON equality preserves boolean/number and integer/float distinctions.
"""
import json
import math


class ConditionError(ValueError):
    def __init__(self, message, status='incomparable'):
        super().__init__(message)
        self.status = status


def require(ok, message, status='incomparable'):
    if not ok:
        raise ConditionError(message, status)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def known(value):
    if value is None or value == '':
        return False
    if isinstance(value, dict):
        return bool(value) and all(known(v) or
            (k == 'target_dtype' and v is None and value.get('enabled') is False)
            for k, v in value.items())
    if isinstance(value, list):
        return all(known(v) for v in value)
    return type(value) in (str, bool, int, float) and not (isinstance(value, float) and not math.isfinite(value))


def expected_conditions(spec, side, required=()):
    common = spec.get('common')
    ranks = spec.get('expected_ranks')
    require(isinstance(common, dict), 'common must be an object')
    require(isinstance(ranks, dict) and bool(ranks), 'contract expected_ranks required', 'not_covered')
    for name, values in ranks.items():
        require(isinstance(name, str) and bool(name.strip()), 'invalid side ID')
        require(isinstance(values, list) and bool(values) and
                all(type(r) is int and r >= 0 for r in values) and len(values) == len(set(values)),
                'invalid contract ranks')
    require(side in ranks, 'side not declared in contract')
    declared = spec.get('side_conditions', {})
    require(isinstance(declared, dict), 'side_conditions must be an object')
    require(not declared or set(declared) == set(ranks), 'side condition inventory mismatch', 'not_covered')
    for values in declared.values():
        require(isinstance(values, dict) and bool(values), 'side conditions must be nonempty objects')
        require(not set(common) & set(values), 'condition declared both common and per-side')
    if declared:
        require(all(set(v) == set(declared[side]) for v in declared.values()), 'per-side condition fields differ')
    expected = {**common, **declared.get(side, {})}
    canonical(expected)
    require(all(k in expected and known(expected[k]) for k in required),
            'missing/unknown execution condition', 'not_covered')
    return expected


def check_actual(spec, side, actual, required=()):
    expected = expected_conditions(spec, side, required)
    require(isinstance(actual, dict), 'rank actual settings required', 'not_covered')
    for name, value in expected.items():
        require(name in actual, 'missing actual condition: '+name, 'not_covered')
        require(canonical(actual[name]) == canonical(value), 'actual condition differs: '+name)
    return expected


def load_json(path):
    """Read finite JSON with unique object keys."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result
    def reject(value):
        raise ConditionError('nonfinite JSON number: '+value)
    with open(path) as stream:
        result = json.load(stream, object_pairs_hook=pairs, parse_constant=reject)
    canonical(result)
    return result


def execution_settings(spec, data, base, read=load_json, require_actual=True):
    """Validate contract-owned rank coverage and rank environment/status records.

    data uses side, common, expected_ranks and ranks:[{rank,environment,status}].
    Returns expected settings and which ranks supplied checked actual settings.
    """
    side = data.get('side')
    expected = expected_conditions(spec, side)
    require(canonical(data.get('common')) == canonical(spec['common']), 'common conditions differ')
    ranks = spec['expected_ranks'][side]
    declared = data.get('expected_ranks')
    require(isinstance(declared, list) and all(type(r) is int and r >= 0 for r in declared)
            and len(declared) == len(set(declared)) and set(declared) == set(ranks),
            'expected ranks differ from contract', 'not_covered')
    records = data.get('ranks')
    require(isinstance(records, list) and all(isinstance(r, dict) and type(r.get('rank')) is int for r in records),
            'rank records required', 'not_covered')
    require(len(records) == len(ranks) and {r['rank'] for r in records} == set(ranks),
            'rank coverage mismatch', 'not_covered')
    checked = []
    for row in records:
        rank = row['rank']
        for kind in ('environment', 'status'):
            doc = read(base / row[kind])
            require(isinstance(doc, dict) and type(doc.get('rank')) is int and doc['rank'] == rank,
                    kind+' rank mismatch')
            if kind == 'status':
                require(doc.get('status') == 'completed', 'rank did not complete', 'execution_failed')
            else:
                require(isinstance(doc.get('versions'), dict) and bool(doc['versions']),
                        'missing runtime versions', 'not_covered')
                if require_actual or 'actual' in doc:
                    check_actual(spec, side, doc.get('actual'))
                    checked.append(rank)
    return dict(expected=expected, actual_ranks=sorted(checked), expected_ranks=sorted(ranks))
