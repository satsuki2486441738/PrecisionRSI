#!/usr/bin/env python3
"""Validate published evaluation/repair histories and evidence links (stdlib only)."""
import datetime
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit


def validate(site):
    data = json.loads((site / 'data/repairs.json').read_text(encoding='utf-8'))
    if data['schema_version'] != 1:
        raise ValueError('Unsupported history schema')
    datetime.date.fromisoformat(data['updated_at'])
    ids = set()
    for repair in data['repairs']:
        identity = repair['id']
        if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', identity) or identity in ids:
            raise ValueError('Repair IDs must be unique stable slugs')
        ids.add(identity)
        datetime.date.fromisoformat(repair['date'])
        for key in ('title', 'status', 'model', 'environment', 'problem', 'cause',
                    'repair', 'stages', 'validation', 'evidence', 'notes'):
            if key not in repair:
                raise ValueError(f'{identity}: missing {key}')
        if repair['status'] not in ('investigating', 'patched', 'validated'):
            raise ValueError(f'{identity}: invalid status')
        for section, keys in {'model': ('name', 'revision'),
                              'environment': ('hardware', 'libraries'),
                              'problem': ('summary', 'trigger', 'expected', 'observed', 'location'),
                              'cause': ('category', 'summary', 'components'),
                              'repair': ('summary', 'commit', 'upstream_status'),
                              'validation': ('scope', 'coverage', 'metrics', 'checks', 'acceptance', 'remaining')}.items():
            for key in keys:
                if key not in repair[section]:
                    raise ValueError(f'{identity}: missing {section}.{key}')
        coverage = repair['validation']['coverage']
        observed, expected = coverage['observed'], coverage['expected']
        if (type(observed) is not int or type(expected) is not int
                or not 0 <= observed <= expected or expected <= 0 or not coverage['unit']):
            raise ValueError(f'{identity}: invalid coverage')
        metric_ids = set()
        for metric in repair['validation']['metrics']:
            if metric['id'] in metric_ids or not metric['label'] or not metric['unit']:
                raise ValueError(f'{identity}: invalid metric identity or unit')
            metric_ids.add(metric['id'])
            for field in ('before', 'after'):
                value = metric[field]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'{identity}: invalid error metric')
        for evidence in repair['evidence']:
            url = urlsplit(evidence['url'])
            if url.scheme:
                if url.scheme != 'https' or not url.netloc:
                    raise ValueError(f'{identity}: invalid evidence URL')
            else:
                target = (site / url.path).resolve()
                if url.netloc or not target.is_relative_to(site.resolve()) or not target.is_file():
                    raise ValueError(f'{identity}: missing or escaped evidence')
    groups = {}
    for group in data.get('library_groups', []):
        identity = group['id']
        if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', identity) or identity in groups:
            raise ValueError('Library group IDs must be unique stable slugs')
        components = group['components']
        if (not group['title'] or not isinstance(components, list) or not components
                or any(not isinstance(name, str) or not name.strip() for name in components)
                or len(set(components)) != len(components)):
            raise ValueError('Library groups need a title and distinct component names')
        groups[identity] = group
    records = {record['id']: record for record in data['repairs']}
    for record in records.values():
        root, seen = record, set()
        while root.get('related_repair_id') is not None:
            if root['id'] in seen:
                raise ValueError('Cyclic repair relationship')
            seen.add(root['id'])
            parent = root['related_repair_id']
            if not isinstance(parent, str) or parent not in records:
                raise ValueError('Missing parent repair')
            root = records[parent]
        group = groups.get(root.get('library_group_id'))
        if group is None:
            raise ValueError('Independent fixes require a declared library group')
        if record.get('library_group_id', group['id']) != group['id']:
            raise ValueError('Follow-up library group differs from its repair')
        if root.get('affected_library') and root['affected_library'] not in group['components']:
            raise ValueError('Affected library must belong to the declared group')
    return len(ids)


def validate_evaluations(site):
    """Check peer identities, measured values, coverage and immutable provenance."""
    data = json.loads((site / 'data/evaluations.json').read_text(encoding='utf-8'))
    if data['schema_version'] != 1:
        raise ValueError('Unsupported evaluation schema')
    datetime.date.fromisoformat(data['updated_at'])
    ids = set()
    def slug(identity, used):
        if not isinstance(identity, str) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', identity) or identity in used:
            raise ValueError('Evaluation identities must be distinct stable slugs')
        used.add(identity)
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    for record in data['evaluations']:
        slug(record['id'], ids)
        datetime.date.fromisoformat(record['date'])
        for name in ('title', 'task', 'status', 'acceptance', 'model', 'summary', 'sides',
                     'common', 'differences', 'comparisons', 'notes', 'evidence', 'archive'):
            if name not in record:
                raise ValueError(f'{record["id"]}: missing {name}')
        if record['task'] not in ('inference', 'sft', 'rl') or record['status'] not in ('completed', 'partial'):
            raise ValueError('Invalid evaluation task/status')
        if not record['title'] or not record['summary'] or not all(record['model'].get(k) for k in ('name', 'revision')):
            raise ValueError('Evaluation needs title, summary and model identity')
        if record['acceptance'] is not None and not isinstance(record['acceptance'], (str, dict)):
            raise ValueError('Invalid numerical acceptance')
        for key in ('common', 'differences', 'notes'):
            if not isinstance(record[key], list) or any(not isinstance(x, str) or not x.strip() for x in record[key]):
                raise ValueError('Evaluation conditions and notes must be text lists')
        sides = set()
        for side in record['sides']:
            slug(side['id'], sides)
            if not side['label'] or not side['hardware'] or not side['libraries'] or not isinstance(side['settings'], dict):
                raise ValueError('Each evaluation side needs its actual configuration')
            if any(not lib.get('name') or not lib.get('version') for lib in side['libraries']):
                raise ValueError('Evaluation libraries need actual names/versions')
        if len(sides) < 2 or not record['comparisons']:
            raise ValueError('Evaluation needs at least two sides and a comparison')
        comparisons = set()
        for comparison in record['comparisons']:
            slug(comparison['id'], comparisons)
            pair = comparison['side_ids']
            if len(pair) != 2 or len(set(pair)) != 2 or not set(pair) <= sides:
                raise ValueError('Comparison must reference two distinct declared sides')
            if not comparison['title'] or not comparison['scope'] or not comparison['metrics'] or not comparison['coverage']:
                raise ValueError('Comparison needs scope, metrics and coverage')
            metrics = set()
            for metric in comparison['metrics']:
                slug(metric['id'], metrics)
                if not metric['label'] or not metric['unit'] or not finite(metric['value']):
                    raise ValueError('Evaluation metrics need a unit and finite numeric value')
                if 'side_values' in metric and (set(metric['side_values']) != set(pair) or
                        not all(finite(v) for v in metric['side_values'].values())):
                    raise ValueError('Metric side values must match the compared sides')
                if 'coordinate' in metric and not isinstance(metric['coordinate'], dict):
                    raise ValueError('Maximum-error coordinate must be an object')
            for coverage in comparison['coverage']:
                observed, expected = coverage['observed'], coverage['expected']
                if (type(observed) is not int or type(expected) is not int or
                        not 0 <= observed <= expected or expected <= 0 or not coverage['label'] or not coverage['unit']):
                    raise ValueError('Invalid evaluation coverage')
        archive = record['archive']
        if (not re.fullmatch('[a-f0-9]{40}', archive['revision']) or not archive['prefix'] or
                type(archive['replay_cases']) is not int or archive['replay_cases'] <= 0):
            raise ValueError('Evaluation archive needs a pinned revision and replay count')
        if not record['evidence']:
            raise ValueError('Evaluation needs published evidence')
        for evidence in record['evidence']:
            url = urlsplit(evidence['url'])
            if not evidence['label']:
                raise ValueError('Evidence needs a label')
            if url.scheme:
                if url.scheme != 'https' or not url.netloc or url.username or url.password:
                    raise ValueError('Invalid evaluation evidence URL')
            else:
                target = (site / url.path).resolve()
                if url.netloc or not target.is_relative_to(site.resolve()) or not target.is_file():
                    raise ValueError('Missing or escaped evaluation evidence')
    return len(ids)


if __name__ == '__main__':
    count = validate(Path(__file__).resolve().parents[1] / 'site')
    evaluations = validate_evaluations(Path(__file__).resolve().parents[1] / 'site')
    print(f'Validated {count} repair verification records and {evaluations} evaluations with evidence links.')
