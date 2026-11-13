#!/usr/bin/env python3
"""Validate survey measurements against their frozen public source payloads."""
import datetime
import json
import math
from pathlib import Path
import re


KINDS = {'cross-backend', 'native-reference', 'within-backend',
         'operator-reference', 'reported-diagnostic'}


def resolve_pointer(value, pointer):
    """Resolve an RFC 6901 pointer, including escaped object keys and list indices."""
    if pointer == '':
        return value
    if not isinstance(pointer, str) or not pointer.startswith('/'):
        raise ValueError('Source pointers must use JSON Pointer syntax')
    try:
        for part in pointer[1:].split('/'):
            part = part.replace('~1', '/').replace('~0', '~')
            value = value[int(part)] if isinstance(value, list) else value[part]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f'Unresolved source pointer: {pointer}') from error
    return value


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_data(data, raw):
    if data.get('schema_version') != 1 or raw.get('schema_version') != 1:
        raise ValueError('Unsupported differences schema')
    datetime.date.fromisoformat(data['updated_at'])
    branches = data['branches']
    if not branches or len({x['branch'] for x in branches}) != len(branches):
        raise ValueError('Survey branch provenance must be distinct')
    for branch in branches:
        if not branch['branch'].startswith('survey/') or not re.fullmatch('[a-f0-9]{40}', branch['commit']):
            raise ValueError('Survey sources must pin immutable commits')
    sources = {source['id']: source for source in data['sources']}
    if len(sources) != len(data['sources']) or not sources:
        raise ValueError('Source identities must be unique')
    if not set(raw['sources']) <= set(sources):
        raise ValueError('Raw payloads must belong to the public source index')
    models = {model['id'] for model in data['models']}
    ids, measurements = set(), 0
    for record in data['records']:
        identity = record['id']
        if not isinstance(identity, str) or not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*', identity) or identity in ids:
            raise ValueError('Difference IDs must be distinct stable slugs')
        ids.add(identity)
        if record['model_id'] not in models or record['comparison_kind'] not in KINDS:
            raise ValueError(f'{identity}: unknown model or comparison kind')
        if record.get('acceptance') is not None:
            raise ValueError(f'{identity}: collection completion must not imply acceptance')
        source = record['source']
        source_id = source['id']
        if source_id not in sources or source_id not in raw['sources']:
            raise ValueError(f'{identity}: undeclared source')
        if not re.fullmatch('[a-f0-9]{40}', source['commit']) or source['commit'] not in source['url']:
            raise ValueError(f'{identity}: evidence links must pin a commit')
        payload = raw['sources'][source_id]
        resolve_pointer(payload, source['json_pointer'])
        if not record['label'] or not record['scope']:
            raise ValueError(f'{identity}: missing measurement scope')
        keys = set()
        for metric in record['metrics']:
            if metric['key'] in keys:
                raise ValueError(f'{identity}: duplicate metric key')
            keys.add(metric['key'])
            if not finite(metric['value']):
                raise ValueError(f'{identity}: nonfinite metric')
            if 'source_pointer' in metric:
                original = resolve_pointer(payload, metric['source_pointer'])
                if not finite(original) or original != metric['value']:
                    raise ValueError(f'{identity}: metric differs from original source')
            elif not metric.get('derivation'):
                raise ValueError(f'{identity}: metric lacks provenance')
            for peer in metric.get('side_values', []):
                if not finite(peer['value']) or not peer.get('label'):
                    raise ValueError(f'{identity}: invalid peer value')
                original = resolve_pointer(payload, peer['source_pointer'])
                if not finite(original) or original != peer['value']:
                    raise ValueError(f'{identity}: peer value differs from source')
            measurements += 1
    if not ids:
        raise ValueError('Survey archive is empty')
    coverage = data['coverage']
    if coverage['recorded_views'] != len(ids) or coverage['numeric_metrics'] != measurements:
        raise ValueError('Published coverage counts do not match the collected views')
    if coverage['branches'] != len(branches) or coverage['unique_source_files'] != len(sources) or coverage['raw_json_sources'] != len(raw['sources']):
        raise ValueError('Published source coverage does not match provenance')
    return len(ids), measurements


def validate(site):
    data = json.loads((site / 'data/differences.json').read_text(encoding='utf-8'))
    raw = json.loads((site / 'data/differences-raw.json').read_text(encoding='utf-8'))
    return validate_data(data, raw)


if __name__ == '__main__':
    records, metrics = validate(Path(__file__).resolve().parents[1] / 'site')
    print(f'Validated {records} survey measurement views and {metrics} source-backed metrics.')
