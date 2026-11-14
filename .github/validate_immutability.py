"""Reject edits or removals of published history relative to a base revision.

Usage: python -B .github/validate_immutability.py [--base REF]

Recorded evaluations, repairs, library groups, survey measurement views and
published evidence files are append-only. A change may add records or files,
and may update top-level ``updated_at``/``snapshot_date`` fields, but every
record identity present at the base revision must remain with identical
content, and every evidence file must keep its bytes and path.
"""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
COLLECTIONS = {
    'site/data/evaluations.json': ('evaluations',),
    'site/data/repairs.json': ('repairs', 'library_groups'),
    'site/data/differences.json': ('records', 'models', 'sources'),
}
EVIDENCE = 'site/evidence/'


def _identity(item):
    return item.get('id') if isinstance(item, dict) else None


def compare_documents(path, base, head, keys):
    """Return violations for records of ``base`` that are missing or changed in ``head``."""
    errors = []
    if base.get('schema_version') != head.get('schema_version'):
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
                if index >= len(new) or new[index] != item:
                    errors.append(f'{path}: {key}[{index}] without an id changed')
                continue
            if identity not in current:
                errors.append(f'{path}: {key} record {identity} was removed')
            elif current[identity] != item:
                errors.append(f'{path}: {key} record {identity} was modified')
    return errors


def compare_evidence(changes):
    """Return violations from ``git diff --name-status`` lines under site/evidence."""
    errors = []
    for line in changes:
        fields = line.split('\t')
        status, paths = fields[0], fields[1:]
        if status.startswith('A'):
            continue
        if any(path.startswith(EVIDENCE) for path in paths):
            errors.append(f'evidence {paths[0]} changed ({status})')
    return errors


def _git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True,
                          text=True, encoding='utf-8').stdout


def validate(base):
    errors = []
    for path, keys in COLLECTIONS.items():
        try:
            old = json.loads(_git('show', f'{base}:{path}'))
        except subprocess.CalledProcessError:
            continue
        head_file = ROOT / path
        if not head_file.exists():
            errors.append(f'{path}: published history file was removed')
            continue
        errors += compare_documents(path, old, json.loads(head_file.read_text(encoding='utf-8')), keys)
    changes = _git('diff', '--no-renames', '--name-status', base, '--', EVIDENCE).splitlines()
    return errors + compare_evidence(changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--base', default='origin/main', help='revision holding the published history')
    base = parser.parse_args().base
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
