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
    return len(ids)


if __name__ == '__main__':
    count = validate(Path(__file__).resolve().parents[1] / 'site')
    print(f'Validated {count} repair history entries and evidence links.')
