"""Reject edits or removals of published history relative to a base revision.

Usage: python -B -m scripts validate-immutability [--base REF | --ci]

Recorded evaluations, repairs, library groups, survey measurement views and
published evidence files are append-only. A change may add records or files,
and may update top-level ``updated_at``/``snapshot_date`` fields, but every
record identity present at the base revision must remain with identical
content, and every evidence file must keep its bytes and site-relative path.
The former root site/ maps explicitly to community/site/; this migration does
not permit changing records, dropping evidence or renaming evidence within site.
The explicitly pinned 2026-11-17 H20Z-to-H200 catalog correction is the only
record-edit exception; it preserves original native evidence and numerical data.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

from community.records.hardware import normalize_hardware_names

ROOT = pathlib.Path(__file__).resolve().parents[1]
COLLECTIONS = {
    'community/site/data/evaluations.json': ('evaluations',),
    'community/site/data/repairs.json': ('repairs', 'library_groups'),
    'community/site/data/relations.json': ('entities', 'links'),
    'community/site/data/differences.json': ('records', 'models', 'sources'),
}
EVIDENCE = 'community/site/evidence/'
LEGACY_SITE = 'site/'
SITE = 'community/site/'
HARDWARE_CORRECTION = EVIDENCE + 'hardware-name-correction-H200-20261117.json'
HARDWARE_CORRECTION_SHA256 = '373500b8c65df6df1c9c02f3dcbc9ce907354fdbc1e93aa58324f197195efa8f'


def _digest(record):
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def hardware_corrections():
    """Load exact before/after identities, never a general permission to edit history."""
    path = ROOT / HARDWARE_CORRECTION
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HARDWARE_CORRECTION_SHA256:
        raise ValueError('hardware-name correction ledger hash mismatch')
    ledger = json.loads(raw)
    return {(entry['file'], entry['collection'], entry['id']):
            (entry['before_sha256'], entry['after_sha256']) for entry in ledger['entries']}


def _approved_correction(path, collection, old, new, corrections):
    pair = (corrections or {}).get((path, collection, _identity(old)))
    return (pair is not None and pair == (_digest(old), _digest(new)) and
            _same_json(normalize_hardware_names(old), new))


def _identity(item):
    return item.get('id') if isinstance(item, dict) else None


def _same_json(left, right):
    """Compare parsed JSON without Python's bool/int/float coercion."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json(a, b) for a, b in zip(left, right))
    return left == right


def compare_documents(path, base, head, keys, corrections=None):
    """Return violations for records of ``base`` that are missing or changed in ``head``."""
    errors = []
    if not _same_json(base.get('schema_version'), head.get('schema_version')):
        errors.append(f'{path}: schema_version changed')
    for key in keys:
        old, new = base.get(key) or [], head.get(key) or []
        if not isinstance(new, list):
            errors.append(f'{path}: {key} is no longer a list')
            continue
        current = {}
        for item in new:
            identity = _identity(item)
            if identity is not None:
                current[identity] = item
        for index, item in enumerate(old):
            identity = _identity(item)
            if identity is None:
                if index >= len(new) or not _same_json(new[index], item):
                    errors.append(f'{path}: {key}[{index}] without an id changed')
                continue
            if identity not in current:
                errors.append(f'{path}: {key} record {identity} was removed')
            elif (not _same_json(current[identity], item) and not
                  _approved_correction(path, key, item, current[identity], corrections)):
                errors.append(f'{path}: {key} record {identity} was modified')
    return errors


def compare_evidence(changes):
    """Return violations from ``git diff --name-status`` lines under community/site/evidence."""
    errors = []
    for line in changes:
        fields = line.split('\t')
        status, paths = fields[0], fields[1:]
        if status.startswith('A'):
            continue
        if any(path.startswith((EVIDENCE, LEGACY_SITE + 'evidence/')) for path in paths):
            errors.append(f'evidence {paths[0]} changed ({status})')
    return errors


def _git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True,
                          text=True, encoding='utf-8').stdout


def validate(base):
    errors = []
    try:
        corrections = hardware_corrections()
    except ValueError as error:
        return [str(error)]
    for path, keys in COLLECTIONS.items():
        for original in (path, LEGACY_SITE + path.removeprefix(SITE)):
            try:
                old = json.loads(_git('show', f'{base}:{original}'))
            except subprocess.CalledProcessError:
                continue
            head_file = ROOT / path
            if not head_file.is_file():
                errors.append(f'{path}: published history file was removed')
                continue
            errors += compare_documents(path, old, json.loads(head_file.read_text(encoding='utf-8')), keys,
                                        corrections=corrections)
    # Compare base blobs against working-tree bytes across the explicit directory
    # migration. Diff rename heuristics cannot prove that evidence was preserved.
    entries = _git('ls-tree', '-r', '-z', base, '--', EVIDENCE,
                   LEGACY_SITE + 'evidence/').split('\0')
    for entry in filter(None, entries):
        metadata, original = entry.split('\t', 1)
        mode, kind, identity = metadata.split()
        current = SITE + original.removeprefix(LEGACY_SITE) if original.startswith(LEGACY_SITE) else original
        target = ROOT / current
        if (mode != '100644' and mode != '100755') or kind != 'blob' or target.is_symlink() or not target.is_file():
            errors.append(f'evidence {current} missing or not a regular file')
        elif _git('hash-object', '--no-filters', str(target)).strip() != identity:
            errors.append(f'evidence {current} changed')
    return errors


def ci_base(event):
    """Protect published history; a draft branch's previous tip is not a release."""
    if event.get('BASE_REF'):
        return 'origin/' + event['BASE_REF']
    before = event.get('BEFORE', '')
    if event.get('GITHUB_REF') == 'refs/heads/main' and before and set(before) != {'0'}:
        try:
            _git('cat-file', '-e', f'{before}^{{commit}}')
        except subprocess.CalledProcessError:
            pass
        else:
            return before
    return 'origin/main'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--base', help='revision holding the published history (default: origin/main)')
    group.add_argument('--ci', action='store_true', help='select the published baseline from CI event variables')
    args = parser.parse_args()
    base = ci_base(os.environ) if args.ci else args.base or 'origin/main'
    try:
        _git('rev-parse', '--verify', f'{base}^{{commit}}')
    except subprocess.CalledProcessError:
        sys.exit(f'Base revision {base!r} is unavailable; fetch it before validating history.')
    errors = validate(base)
    if errors:
        sys.exit('Published history must stay append-only:\n' + '\n'.join(errors))
    print(f'Published history is append-only with the pinned H200 naming correction relative to {base}.')


if __name__ == '__main__':
    main()
