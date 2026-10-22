#!/usr/bin/env python3
"""Register and find verified reusable environments, models, adapters or captures.

API: register(catalog_path, entry_path) -> receipt; find(catalog_path, kind=None,
properties=None, limit=5) -> current matching resources. Standard library, POSIX.
CLI: resource_catalog.py register --catalog C --entry E.json
     resource_catalog.py find --catalog C [--kind environment|model|adapter|capture]
                             [--match '{"framework":"torch"}'] [--limit 5]

Entry JSON: id (alphanumeric/hyphens), kind, properties:{...},
paths:{name:local_file_or_directory,...}, evidence:record_run_command_json.
Relative paths resolve against the entry file. Evidence must record a completed
native command with returncode=0. The caller declares the verified support scope;
this receipt does not assert correctness or compatibility beyond that scope.
Paths should identify the prepared resource and relevant model/data/source/runtime
identity files. Remote resources use their locally verified identity receipts.
Only declared content is checked; use a live probe when remote/runtime identity
cannot be established from retained files. Native execution remains task-owned.

Registration fingerprints paths and evidence; identical entries are no-ops.
New versions append, preserving earlier entries. Find uses exact supplied property
matches, validates current selected identities, and returns only each ID's latest
valid registration. Missing/changed files are excluded. Returned paths can be
used as workflow inputs or native command arguments; numeric arrays stay in place.
Find performs no native command, model call or file creation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from conditions import canonical, load_json
from workflow import fingerprint, workspace_lock

KINDS = {'environment', 'model', 'adapter', 'capture'}
ID = re.compile(r'[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\Z')


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _read(catalog):
    if not catalog.exists():
        return dict(schema_version=1, versions=[])
    result = load_json(catalog)
    _require(isinstance(result, dict) and result.get('schema_version') == 1
             and isinstance(result.get('versions'), list), 'invalid resource catalog')
    return result


def register(catalog_path, entry_path):
    """Register task-verified resources with content identities and native evidence."""
    source = Path(entry_path).resolve()
    spec = load_json(source)
    _require(isinstance(spec, dict) and isinstance(spec.get('id'), str) and ID.fullmatch(spec['id']), 'resource id required')
    _require(spec.get('kind') in KINDS, 'unknown resource kind')
    properties = spec.get('properties', {})
    _require(isinstance(properties, dict), 'properties must be an object')
    paths = spec.get('paths')
    _require(isinstance(paths, dict) and paths and all(isinstance(k, str) and ID.fullmatch(k)
             and isinstance(v, str) and v for k, v in paths.items()), 'named resource paths required')
    _require(isinstance(spec.get('evidence'), str) and spec['evidence'], 'native command evidence required')
    evidence = (source.parent / spec['evidence']).resolve()
    receipt = load_json(evidence)
    _require(isinstance(receipt, dict) and receipt.get('status') == 'completed'
             and type(receipt.get('returncode')) is int and receipt['returncode'] == 0
             and isinstance(receipt.get('argv'), list) and receipt['argv'], 'evidence must be a successful native command receipt')
    selected = {}
    for key, value in paths.items():
        path = (source.parent / value).absolute()
        selected[key] = dict(path=str(path), identity=fingerprint(path))
    record = dict(id=spec['id'], kind=spec['kind'], properties=properties, paths=selected,
                  evidence=dict(path=str(evidence), identity=fingerprint(evidence)))
    catalog = Path(catalog_path).resolve()
    catalog.parent.mkdir(parents=True, exist_ok=True)
    with workspace_lock(catalog.parent):
        state = _read(catalog)
        latest = next((r for r in reversed(state['versions']) if r['id'] == record['id']), None)
        if latest and all(canonical(latest[k]) == canonical(v) for k, v in record.items()):
            return dict(id=record['id'], registered=False, resource=latest)
        record['registered_at'] = datetime.now(timezone.utc).isoformat()
        state['versions'].append(record)
        temporary = catalog.with_name(catalog.name + '.tmp')
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
        temporary.replace(catalog)
    return dict(id=record['id'], registered=True, resource=record)


def find(catalog_path, kind=None, properties=None, limit=5):
    """Return bounded exact-property matches after verifying selected content identities."""
    _require(kind is None or kind in KINDS, 'unknown resource kind')
    _require(type(limit) is int and 1 <= limit <= 100, 'limit must be 1..100')
    properties = {} if properties is None else properties
    _require(isinstance(properties, dict), 'match must be an object')
    state = _read(Path(catalog_path).resolve())
    results, seen = [], set()
    for record in reversed(state['versions']):
        if record['id'] in seen:
            continue
        seen.add(record['id'])
        if kind is not None and kind != record['kind']:
            continue
        if any(k not in record['properties'] or canonical(record['properties'][k]) != canonical(v)
               for k, v in properties.items()):
            continue
        try:
            valid = all(fingerprint(v['path']) == v['identity']
                        for v in [*record['paths'].values(), record['evidence']])
        except (OSError, ValueError):
            valid = False
        if valid:
            results.append(record)
            if len(results) == limit:
                break
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('register', 'find'))
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--entry')
    parser.add_argument('--kind', choices=sorted(KINDS))
    parser.add_argument('--match', default='{}')
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args(argv)
    try:
        if args.command == 'register':
            _require(args.entry is not None, '--entry required')
            result = register(args.catalog, args.entry)
        else:
            result = dict(resources=find(args.catalog, args.kind, json.loads(args.match), args.limit))
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        print(json.dumps(dict(status='error', reason=str(exc))))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
