"""Prepare and publish append-only community drafts from an isolated Git checkout."""

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


UPSTREAM = 'satsuki2486441738/NumericBridge'
SITE = 'community/site/'
REPOSITORY = re.compile(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}')
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
        action = ' '.join(argv[:3]) if argv[0] == 'gh' else argv[0]
        raise ValueError(f'{action} failed; check credentials, repository state and connectivity') from error


def git(cwd, *args, **kwargs):
    return run(cwd, 'git', '-c', 'core.hooksPath=/dev/null', *args, **kwargs)


class Submissions:
    def __init__(self, repo, drafts, fork, *, base_ref='refs/remotes/origin/main'):
        if not REPOSITORY.fullmatch(fork) or fork.lower() == UPSTREAM.lower():
            raise ValueError('Configure a contributor-owned GitHub fork as OWNER/REPO')
        self.repo = Path(repo).resolve()
        self.drafts = Path(drafts).resolve()
        self.fork = fork
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

    def prepare(self, title, additions):
        if not any(items for name in COLLECTIONS for items in additions[name].values()):
            raise ValueError('Include at least one community record or relation')
        if len(encode(additions, compact=True)) > 256 * 1024:
            raise ValueError('Limit a submission to 256 KiB of reviewed public JSON')
        base = git(self.repo, 'rev-parse', '--verify', self.base_ref + '^{commit}')
        with self.checkout(base) as root:
            files = self.apply(root, additions)
        preview = dict(schema_version=1, title=title, upstream=UPSTREAM, base_branch='main',
                       fork=self.fork, base_commit=base, additions=additions, files=files)
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
        if (reviewed['fork'], reviewed['upstream'], reviewed['base_branch']) != (self.fork, UPSTREAM, 'main'):
            raise ValueError('Draft destination changed; prepare and review a new draft')
        return reviewed

    def publish(self, draft_id, digest):
        draft = self.reviewed(draft_id, digest)
        with self.checkout(draft['base_commit']) as root:
            if self.apply(root, draft['additions']) != draft['files']:
                raise ValueError('Generated files differ from the reviewed preview')
            return publish_github(root, draft, digest)


def publish_github(root, draft, digest):
    """Publish a digest-addressed commit to the submitter's fork and open one PR."""
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GH_PROMPT_DISABLED': '1'}

    def api(endpoint, *flags, payload=None):
        return json.loads(run(root, 'gh', 'api', '--hostname', 'github.com', *flags,
                              endpoint, payload=payload, env=env))

    actor = api('user')
    login = actor['login']
    fork = draft['fork']
    fork_info = api(f'repos/{fork}')
    if (fork.split('/')[0].lower() != login.lower() or not fork_info['fork'] or
            fork_info.get('source', {}).get('full_name', '').lower() != UPSTREAM.lower()):
        raise ValueError('Sign in as the owner of a personal fork of NumericBridge')

    branch = f'numericbridge/submission-{digest}'
    ref = f'refs/heads/{branch}'
    base = draft['base_commit']
    git(root, 'add', '--', *draft['files'])
    for path in draft['files']:
        if git(root, 'rev-parse', f':{path}') != git(root, 'hash-object', '--no-filters', '--', path):
            raise ValueError('Git staging changed the reviewed bytes: ' + path)
    tree = git(root, 'write-tree')
    date = git(root, 'show', '-s', '--format=%cI', base)
    email = f"{actor['id']}+{login}@users.noreply.github.com"
    commit_env = {**env, 'GIT_AUTHOR_NAME': login, 'GIT_COMMITTER_NAME': login,
                  'GIT_AUTHOR_EMAIL': email, 'GIT_COMMITTER_EMAIL': email,
                  'GIT_AUTHOR_DATE': date, 'GIT_COMMITTER_DATE': date}
    commit = git(root, '-c', 'commit.gpgsign=false', 'commit-tree', tree, '-p', base,
                 '-m', draft['title'], env=commit_env)

    def previous_pull():
        items = api(f'repos/{UPSTREAM}/pulls', '--method', 'GET', '-f', 'state=all',
                    '-f', f'head={login}:{branch}', '-f', 'base=main')
        if not items:
            return None
        if len(items) != 1 or items[0]['head']['sha'] != commit:
            raise ValueError('An existing pull request differs from this reviewed draft')
        return items[0]['html_url']

    previous = previous_pull()
    if previous:
        return dict(status='submitted', pull_request_url=previous, reused=True)
    if api(f'repos/{UPSTREAM}/git/ref/heads/main')['object']['sha'] != base:
        return dict(status='stale_base', message='Refresh origin/main and review a new draft')

    def fork_head():
        refs = api(f'repos/{fork}/git/matching-refs/{ref.removeprefix("refs/")}')
        return next((item['object']['sha'] for item in refs if item['ref'] == ref), None)

    head = fork_head()
    if head is None:
        try:
            git(root, '-c', 'credential.helper=', '-c', 'credential.helper=!gh auth git-credential',
                'push', f'--force-with-lease={ref}:', f'https://github.com/{fork}.git',
                f'{commit}:{ref}', env=env)
        except ValueError:
            if fork_head() != commit:
                raise
    elif head != commit:
        raise ValueError('The existing fork branch differs from this reviewed draft')
    body = dict(title=draft['title'], head=f'{login}:{branch}', base='main',
                body=f'NumericBridge community contribution.\n\nReviewed draft SHA-256: `{digest}`.',
                maintainer_can_modify=False)
    try:
        result = api(f'repos/{UPSTREAM}/pulls', '--method', 'POST', '--input', '-',
                     payload=encode(body).decode('utf-8'))
    except ValueError:
        previous = previous_pull()
        if previous:
            return dict(status='submitted', pull_request_url=previous, reused=True)
        raise
    if result['head']['sha'] != commit:
        raise ValueError('GitHub returned a different pull request head; inspect the contributor fork')
    return dict(status='submitted', pull_request_url=result['html_url'], reused=False)
