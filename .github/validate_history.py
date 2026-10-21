#!/usr/bin/env python3
"""Validate the published repair history and its local evidence links (stdlib only)."""
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


if __name__ == '__main__':
    count = validate(Path(__file__).resolve().parents[1] / 'site')
    print(f'Validated {count} verification records, library groups and evidence links.')
