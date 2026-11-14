"""Export new evaluation statistics, retaining scope without copying private artifacts.

Usage: python -B .github/export_statistics.py --base SITE/data/evaluations.json
       --source OTHER_SITE/data/evaluations.json --output NEW_FILE.json

Output is a separate statistics-only evidence bundle, not a replacement history
catalog. It contains no source archive URLs, local refs, arrays or replay claims.
Manual publication review is still required; a text scan cannot certify privacy.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

SPEC = importlib.util.spec_from_file_location('history_integration',
    Path(__file__).with_name('plan_history_integration.py'))
INTEGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INTEGRATION)
FIELDS = {'id', 'date', 'title', 'task', 'status', 'scope_status', 'whole_task_completed',
          'acceptance', 'model', 'summary', 'sides', 'common', 'differences',
          'comparisons', 'notes', 'visibility'}
PRIVATE = re.compile(r'(?i)(?:/Users/|/home/|\bssh\b|\blion\b|\bmodelmate\d*\b|'
                     r'\b(?:PID|PGID|JobUID|PodUID|boot_id|pidbirth)\b|refs/precision-archives/)')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def display(value):
    """Apply the owner's confirmed hardware display name only to string values."""
    if isinstance(value, str):
        return value.replace('H20Z', 'H200')
    if isinstance(value, list):
        return [display(item) for item in value]
    if isinstance(value, dict):
        return {key: display(item) for key, item in value.items()}
    return value


def export(base, source):
    base_path, base_data, _ = INTEGRATION.load(base)
    source_path, source_data, source_hash = INTEGRATION.load(source)
    if base_data.get('schema_version') != source_data.get('schema_version'):
        raise ValueError('schema_version differs from base')
    existing = {r['id']: r for r in base_data['evaluations']}
    records = []
    for original in source_data['evaluations']:
        identity = original['id']
        if identity in existing:
            if canonical(original) != canonical(existing[identity]):
                raise ValueError(f'conflicting existing evaluation: {identity}')
            continue
        unknown = set(original) - FIELDS - {'archive', 'evidence'}
        if unknown:
            raise ValueError(f'{identity}: unreviewed fields {sorted(unknown)}')
        record = display({key: deepcopy(value) for key, value in original.items() if key in FIELDS})
        record['provenance'] = {
            'source_record_sha256': hashlib.sha256(canonical(original)).hexdigest(),
            'raw_artifacts_public': False, 'independent_replay_provided': False,
            'hardware_display_normalized': True}
        scope = original.get('archive', {}).get('replay_cases_scope')
        if scope:
            record['provenance']['retained_source_scope'] = display(scope)
        if PRIVATE.search(json.dumps(record, ensure_ascii=False)):
            raise ValueError(f'{identity}: potential private execution detail requires review')
        records.append(record)
    return {'schema_version': 1, 'kind': 'statistics-only',
            'source_snapshot_date': source_data['updated_at'],
            'source_catalog_sha256': source_hash,
            'source_access': 'Public reachability is not established by this export.',
            'scope': [
                'Historical evaluation snapshots; completion and acceptance retain their original scope.',
                'Statistics and declared coverage only; original arrays and execution traces remain private.',
                'No independently downloadable numerical replay is provided by this bundle.',
                'Comparison differences do not establish implementation defects or hardware causality.',
                'Historical partial records are not updated using later experiments.'],
            'evaluations': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = export(args.base, args.source)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({'evaluations': len(result['evaluations']), 'kind': result['kind'],
                      'manual_publication_review_required': True}))


if __name__ == '__main__':
    main()
