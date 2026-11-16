#!/usr/bin/env python3
"""Project retained results into public metadata without local archive downloads.

CLI: --input JSON --archives local-evidence.json --output NEW_JSON.
Archive metadata retains names, hashes and availability. Measurements, IDs,
coverage and acceptance are preserved. This does not replay native mathematics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()


def archive_name(value, archives):
    if not isinstance(value, str):
        return None
    name = Path(urlsplit(value).path).name
    return name if name in archives else None


def report_text(text, archives):
    def replace(match):
        return (match[2]+' [local evidence retained; not published]'
                if archive_name(match[3], archives) else match[0])
    return re.sub(r'(!?)\[([^\]]*)\]\(([^)]+)\)', replace, text)


def project(value, archives):
    """Return a public copy; never alter numeric observations or source IDs."""
    if isinstance(value, list):
        return [project(item, archives) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    retained = []
    for key, item in value.items():
        if key == 'evidence' and isinstance(item, list):
            public = []
            for evidence in item:
                name = archive_name(evidence.get('url'), archives) if isinstance(evidence, dict) else None
                if name:
                    retained.append(dict(label=evidence.get('label', name), **archives[name]))
                else:
                    public.append(project(evidence, archives))
            result[key] = public
        elif key in ('url', 'download', 'href') and archive_name(item, archives):
            result['local_archive'] = dict(archives[archive_name(item, archives)])
        elif key == 'text' and isinstance(item, str):
            result[key] = report_text(item, archives)
        else:
            result[key] = project(item, archives)
    if retained:
        result['local_evidence'] = result.get('local_evidence', []) + retained
    if result.get('kind') == 'portable_supplement' and archive_name(result.get('file'), archives):
        name = archive_name(result['file'], archives)
        result.update(kind='local_archive', availability='local-only', file=name)
    # This hash describes the current public source payload; source_sha256
    # continues to identify the original retained source bytes.
    if 'export_sha256' in result and isinstance(result.get('data'), (dict, list)):
        result['export_sha256'] = hashlib.sha256(encoded(result['data'])).hexdigest()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be a new file')
    archives = {item['name']: item for item in json.loads(args.archives.read_text())['archives']}
    args.output.write_bytes(encoded(project(json.loads(args.input.read_text()), archives)))
