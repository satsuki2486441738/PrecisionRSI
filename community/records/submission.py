"""Prepare and verify append-only community drafts from an isolated Git checkout."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from scripts.validate_immutability import compare_documents
from .catalog import Catalog
from .search import compile_index
from .validation import validate, validate_evaluations


SITE = 'community/site/'
DIGEST = re.compile(r'[a-f0-9]{64}')
EVIDENCE = re.compile(r'evidence/submissions/[a-z0-9]+(?:-[a-z0-9]+)*\.json')
COLLECTIONS = {'evaluations': ('evaluations',), 'repairs': ('repairs', 'library_groups'),
               'relations': ('entities', 'links')}


def encode(value, *, compact=False):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                       separators=(',', ':') if compact else None,
                       indent=None if compact else 2) + '\n').encode('utf-8')


def run(cwd, *argv, payload=None, env=None):
    try:
        return subprocess.run(argv, cwd=cwd, input=payload, text=True, encoding='utf-8',
                              env=env, capture_output=True, check=True, timeout=180).stdout.strip()
    except FileNotFoundError as error:
        raise ValueError(f'Required command unavailable: {argv[0]}') from error
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(f'{argv[0]} failed while preparing the community preview') from error


def git(cwd, *args, **kwargs):
    return run(cwd, 'git', '-c', 'core.hooksPath=/dev/null', *args, **kwargs)


class Submissions:
    def __init__(self, repo, drafts, *, destination=None, base_ref='refs/remotes/origin/main'):
        self.repo = Path(repo).resolve()
        self.drafts = Path(drafts).resolve()
        self.destination = destination
        self.base_ref = base_ref

    @contextmanager
    def checkout(self, base):
        with TemporaryDirectory(prefix='numericbridge-submission-') as temporary:
            root = Path(temporary).resolve() / 'checkout'
            git(self.repo, 'clone', '--quiet', '--no-checkout', '--shared', '--', str(self.repo), str(root))
            git(root, 'checkout', '--quiet', '--detach', base)
            yield root

    def apply(self, root, additions):
        site = root / SITE
        changed = []
        for path, document in additions['evidence'].items():
            if len(path) > 160 or not EVIDENCE.fullmatch(path):
                raise ValueError('New evidence must be a JSON file under evidence/submissions/')
            target = site / path
            if target.resolve() != target:
                raise ValueError('Submission evidence paths must be regular files in the review checkout')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(encode(document))
            changed.append(SITE + path)
        for name, keys in COLLECTIONS.items():
            values = additions[name]
            if not any(values.values()):
                continue
            path = f'{SITE}data/{name}.json'
            raw = (root / path).read_bytes()
            old = json.loads(raw)
            new = old.copy()
            for key, items in values.items():
                if items:
                    new[key] = [*old.get(key, []), *items]
            if 'updated_at' in old:
                new['updated_at'] = max([old['updated_at'], *(item['date'] for item in values.get(name, []))])
            violations = compare_documents(path, old, new, keys)
            if violations:
                raise ValueError('; '.join(violations))
            (root / path).write_bytes(encode(new, compact=len(raw.splitlines()) == 1))
            changed.append(path)
        validate(site)
        validate_evaluations(site)
        index = f'{SITE}data/search-index.json'
        (root / index).write_bytes(encode(compile_index(Catalog(site)), compact=True))
        changed.append(index)
        return {path: sha256((root / path).read_bytes()).hexdigest() for path in sorted(changed)}

    def preview(self, title, additions):
        if not isinstance(title, str) or not re.fullmatch(r'[^\x00-\x1f\x7f]{1,120}', title.strip()):
            raise ValueError('Use a single-line contribution title of at most 120 characters')
        if not any(items for name in COLLECTIONS for items in additions[name].values()):
            raise ValueError('Include at least one community record or relation')
        if len(encode(additions, compact=True)) > 256 * 1024:
            raise ValueError('Limit a submission to 256 KiB of reviewed public JSON')
        base = git(self.repo, 'rev-parse', '--verify', self.base_ref + '^{commit}')
        with self.checkout(base) as root:
            files = self.apply(root, additions)
        return dict(schema_version=2, title=title.strip(), destination=self.destination,
                    base_commit=base, additions=additions, files=files)

    def prepare(self, title, additions):
        preview = self.preview(title, additions)
        raw = encode(preview, compact=True)
        digest = sha256(raw).hexdigest()
        self.drafts.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.drafts / f'draft-{digest}.json'
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ValueError('Stored draft differs from its preview')
        else:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw)
        return dict(status='ready', draft_id=f'draft-{digest}', draft_sha256=digest, preview=preview)

    def reviewed(self, draft_id, digest):
        if not DIGEST.fullmatch(digest) or draft_id != f'draft-{digest}':
            raise ValueError('Draft ID and SHA-256 must match the exact reviewed preview')
        raw = (self.drafts / f'{draft_id}.json').read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError('Stored draft changed; prepare and review a new draft')
        reviewed = json.loads(raw)
        if reviewed.get('schema_version') != 2 or reviewed.get('destination') != self.destination:
            raise ValueError('Draft destination changed; prepare and review a new draft')
        return reviewed


def main():
    """Validate a local preview using title and additions from a JSON file."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new local JSON preview file')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        with args.input.open('rb') as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError('Local preview input must not exceed 1 MiB')
        request = json.loads(raw)
        preview = Submissions(root, args.output.parent).preview(request['title'], request['additions'])
        data = encode(dict(status='local_draft', preview=preview))
        with os.fdopen(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
            stream.write(data)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, RecursionError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
