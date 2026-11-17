"""Reject edits or removals of published history relative to a base revision.

Usage: python -B -m precision_maintenance validate-immutability [--base REF | --ci]

Recorded evaluations, repairs, library groups, survey measurement views and
published evidence files are append-only. A change may add records or files,
and may update top-level ``updated_at``/``snapshot_date`` fields, but every
record identity present at the base revision must remain with identical
content, and every evidence file must keep its bytes and site-relative path.
The former root site/ maps explicitly to community/site/; this migration does
not permit changing records, dropping evidence or renaming evidence within site.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

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


def compare_documents(path, base, head, keys):
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
            elif not _same_json(current[identity], item):
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
            errors += compare_documents(path, old, json.loads(head_file.read_text(encoding='utf-8')), keys)
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
    print(f'Published history is append-only relative to {base}.')


if __name__ == '__main__':
    main()
