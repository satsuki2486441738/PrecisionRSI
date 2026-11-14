"""Plan an append-only evaluation union without publishing or changing source files.

Usage: python -B .github/plan_history_integration.py --base SITE/data/evaluations.json
       --source NAME=OTHER_SITE/data/evaluations.json [--source NAME=PATH ...]
       --output NEW_PRIVATE_DIRECTORY

Each input uses its own parent SITE directory for evidence resolution. Output is
private: audit.json contains input paths, hashes, source memberships and review
findings; candidate.json is written only when there are no conflicting records.
This is a planning tool, not a privacy certification or publication command.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

MAX_PUBLIC_BYTES = 50 * 1024 * 1024
SCAN_BYTES = 8 * 1024 * 1024
RUNTIME = re.compile(r'(?i)(?:job[_-]?uid|pod[_-]?uid|boot[_-]?id|pidbirth|/Users/|/home/|ssh\s+lion\b)')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load(path):
    path = Path(path).resolve()
    # Reject duplicate JSON keys and non-finite metrics rather than normalizing them.
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'{path}: duplicate JSON key {key!r}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'{path}: non-finite JSON value {value}')
    def finite(value):
        result = float(value)
        if not math.isfinite(result):
            return invalid(value)
        return result
    raw = path.read_bytes()
    data = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                      parse_constant=invalid, parse_float=finite)
    if not isinstance(data, dict) or not isinstance(data.get('evaluations'), list):
        raise ValueError(f'{path}: expected an evaluations list')
    seen = set()
    for record in data['evaluations']:
        identity = record.get('id') if isinstance(record, dict) else None
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError(f'{path}: missing or duplicate evaluation id {identity!r}')
        seen.add(identity)
    return path, data, hashlib.sha256(raw).hexdigest()


def evidence_findings(record, site):
    """Inspect direct references only; never unpack archives or fetch external URLs."""
    site = Path(site).resolve()
    findings, files = [], []
    archive = record.get('archive') or {}
    if archive.get('pushed') is False or archive.get('published') is False or archive.get('local_ref'):
        findings.append('private_archive_provenance')
    elif archive:
        findings.append('archive_revision_public_reachability_unverified')
    if 'H20Z' in json.dumps(record, ensure_ascii=False):
        findings.append('hardware_display_alias_requires_H200')
    if RUNTIME.search(json.dumps(record, ensure_ascii=False)):
        findings.append('record_runtime_identifier_review')
    if not record.get('evidence'):
        findings.append('missing_evidence_references')
    for item in record.get('evidence', []):
        url = item.get('url') if isinstance(item, dict) else None
        if not isinstance(url, str) or not url:
            findings.append('invalid_evidence_reference')
            continue
        parts = urlsplit(url)
        entry = {'url': url, 'findings': []}
        files.append(entry)
        if parts.scheme in ('http', 'https') and parts.netloc:
            entry['findings'].append('external_evidence_unverified')
            continue
        relative = Path(unquote(parts.path))
        target = (site / relative).resolve()
        if parts.scheme or parts.netloc or relative.is_absolute() or not target.is_relative_to(site):
            entry['findings'].append('unsafe_evidence_path')
            continue
        if not target.is_file():
            entry['findings'].append('missing_evidence_file')
            continue
        size = target.stat().st_size
        entry.update(bytes=size, sha256=digest(target))
        if size > MAX_PUBLIC_BYTES:
            entry['findings'].append('exceeds_public_size_limit')
        if target.suffix.lower() in ('.zip', '.gz', '.tar', '.npz', '.npy'):
            entry['findings'].append('raw_or_archive_content_not_reviewed')
        elif size > SCAN_BYTES:
            entry['findings'].append('text_content_exceeds_scan_limit')
        else:
            content = target.read_bytes().decode('utf-8', errors='replace')
            if RUNTIME.search(content):
                entry['findings'].append('evidence_runtime_identifier_review')
    return {'findings': findings, 'evidence': files,
            'publication_review_required': True}


def plan(base, sources):
    """Return (candidate-or-None, private audit); never resolve a conflict by precedence."""
    loaded = [('base', *load(base))]
    names = {'base'}
    for name, path in sources:
        if not name or name in names:
            raise ValueError(f'duplicate or reserved source name {name!r}')
        names.add(name)
        loaded.append((name, *load(path)))
    base_data = loaded[0][2]
    records, origins, conflicts, additions = {}, {}, [], []
    schema = base_data.get('schema_version')
    for name, path, data, _ in loaded:
        if data.get('schema_version') != schema:
            raise ValueError(f'{name}: schema_version differs from base')
        for record in data['evaluations']:
            identity = record['id']
            origins.setdefault(identity, []).append(name)
            if identity in records:
                # JSON equality treats 1/True and 1/1.0 alike; serialized form does not.
                if json.dumps(record, sort_keys=True, ensure_ascii=False) != json.dumps(
                        records[identity], sort_keys=True, ensure_ascii=False):
                    conflicts.append({'id': identity, 'source': name,
                                      'first_source': origins[identity][0]})
                continue
            records[identity] = record
            if name != 'base':
                additions.append({'id': identity, 'source': name,
                                  **evidence_findings(record, path.parent.parent)})
    candidate = deepcopy(base_data)
    candidate['evaluations'] = list(records.values())
    # Preserve each record's historical date and the base metadata; this is not a publication.
    audit = {'schema_version': 1, 'purpose': 'private-integration-plan',
             'publication_authorized_by_tool': False,
             'inputs': [{'name': name, 'path': str(path), 'sha256': source_hash,
                         'records': len(data['evaluations'])}
                        for name, path, data, source_hash in loaded],
             'base_records': len(base_data['evaluations']), 'union_records': len(records),
             'added_records': len(additions), 'memberships': origins,
             'conflicts': conflicts, 'additions': additions,
             'limits': {'direct_evidence_only': True, 'archives_unpacked': False,
                        'external_urls_fetched': False, 'privacy_certified': False,
                        'max_public_bytes': MAX_PUBLIC_BYTES, 'text_scan_bytes': SCAN_BYTES}}
    return (None if conflicts else candidate), audit


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--source', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sources = []
    for item in args.source:
        name, separator, path = item.partition('=')
        if not separator or not path:
            parser.error('--source requires NAME=PATH')
        sources.append((name, Path(path)))
    output = args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo / 'site'):
        parser.error('output is private; choose a fresh directory outside site/')
    try:
        candidate, audit = plan(args.base, sources)
        output.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    def write(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                              allow_nan=False) + '\n', encoding='utf-8')
    write('audit.json', audit)
    if candidate is not None:
        write('candidate.json', candidate)
    print(json.dumps({'base_records': audit['base_records'], 'union_records': audit['union_records'],
                      'added_records': audit['added_records'], 'conflicts': len(audit['conflicts']),
                      'publication_review_required': True}))
    return 2 if candidate is None else 0


if __name__ == '__main__':
    raise SystemExit(main())
