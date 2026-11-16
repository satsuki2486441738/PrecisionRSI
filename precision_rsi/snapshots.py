"""Private immutable research snapshots; inspect, verify and restore with no job replay.

Journal.capture(key, boundary, files={"layer/name": Path}, values={...}, parent=None)
stores exact bytes in SHA256 objects. Layers: platform, changes, session, audit.
Only initial/committed boundaries are checkpoints; submitted/running are evidence.
restore(..., view="researcher") exports only explicitly declared changes/session.
Resource references are identity declarations, not local backups of model weights.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

from .memory import fingerprint, read, workspace_lock


LAYERS = {'platform', 'changes', 'session', 'audit'}
BOUNDARIES = {'initial', 'running', 'submitted', 'committed'}


def _encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _member(name):
    parts = PurePosixPath(name).parts
    if (not parts or parts[0] not in LAYERS or '\\' in name or
            any(p in ('', '.', '..') for p in name.split('/'))):
        raise ValueError('invalid snapshot member')
    if any(p in ('.ssh', '.git', '__pycache__', 'harness-state') or p.startswith(('.env', '.dsh-'))
           or p in ('credentials.json', 'auth.json', 'id_rsa', 'id_ed25519')
           or p.endswith(('.pem', '.key')) for p in parts):
        raise ValueError('credential/runtime member excluded')
    return name


def _identity(identity):
    if not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{64}', identity):
        raise ValueError('invalid snapshot identity')
    return identity


def _immutable(path, raw):
    """Publish once without overwriting another writer; flush bytes before linking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        temporary = Path(out.name)
        try:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.is_symlink() or path.read_bytes() != raw:
                    raise ValueError('immutable snapshot changed')
        finally:
            temporary.unlink(missing_ok=True)


def _platform():
    root = Path(__file__).resolve().parents[1]
    files = {}
    for folder in ('precision_rsi', 'skills'):
        for source in sorted((root/folder).rglob('*')):
            if '__pycache__' not in source.parts and source.suffix != '.pyc' and source.is_file():
                files['platform/'+source.relative_to(root).as_posix()] = source
    for name in ('AGENTS.md', '.github/requirements.txt'):
        if (root/name).is_file():
            files['platform/'+name] = root/name
    return files


class Journal:
    def __init__(self, directory):
        self.root = Path(directory).absolute()
        if self.root.is_symlink():
            raise ValueError('snapshot journal cannot be a symlink')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in ('objects', 'events'):
            folder = self.root/name
            if folder.is_symlink():
                raise ValueError('snapshot storage cannot be a symlink')
            folder.mkdir(exist_ok=True, mode=0o700)

    def _blob(self, raw):
        identity = _hash(raw)
        _immutable(self.root/'objects'/identity, raw)
        return dict(object=identity, bytes=len(raw), executable=False)

    def _file(self, source):
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError('snapshot source must be a regular file')
        before = source.stat()
        with tempfile.NamedTemporaryFile(dir=self.root/'objects', delete=False) as out:
            temporary = Path(out.name)
            try:
                digest = hashlib.sha256()
                with source.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                        out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
                after = source.stat()
                stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode)
                if stamp(before) != stamp(after):
                    raise ValueError('snapshot source changed during capture')
                identity = digest.hexdigest()
                target = self.root/'objects'/identity
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    if target.is_symlink() or fingerprint(target)['sha256'] != identity:
                        raise ValueError('corrupt snapshot object')
            finally:
                temporary.unlink(missing_ok=True)
        return dict(object=identity, bytes=after.st_size, executable=bool(after.st_mode & 0o111))

    def event(self, key):
        if not re.fullmatch('[A-Za-z0-9_-]+', key):
            raise ValueError('invalid snapshot key')
        path = self.root/'events'/(key+'.json')
        recorded = read(path)
        verified = self.verify(recorded['id'])
        if recorded != verified or verified['key'] != key:
            raise ValueError('corrupt snapshot event')
        return verified

    def capture(self, key, boundary, *, files=None, values=None, parent=None, expected=None, directories=()):
        if boundary not in BOUNDARIES or not re.fullmatch('[A-Za-z0-9_-]+', key):
            raise ValueError('invalid snapshot boundary/key')
        with workspace_lock(self.root):
            target = self.root/'events'/(key+'.json')
            # Replay is keyed by an immutable controller boundary, not live files.
            if target.exists():
                prior = self.event(key)
                if prior['boundary'] != boundary or prior['parent'] != parent:
                    raise ValueError('snapshot boundary already bound')
                return prior
            if parent:
                self.verify(parent)
            entries = {}
            for name, source in sorted((files or {}).items()):
                entries[_member(name)] = self._file(source)
                if name in (expected or {}) and entries[name]['object'] != expected[name]:
                    raise ValueError('snapshot artifact identity changed')
            for name, value in sorted((values or {}).items()):
                if name in entries:
                    raise ValueError('duplicate snapshot member')
                entries[_member(name)] = self._blob(_encoded(value))
            # Reject file/directory collisions before publishing the boundary.
            for name in entries:
                if any(str(p) in entries for p in PurePosixPath(name).parents):
                    raise ValueError('snapshot member collision')
            directories = sorted({_member(name) for name in directories})
            if any(name in entries or any(str(p) in entries for p in PurePosixPath(name).parents)
                   for name in directories):
                raise ValueError('snapshot directory collision')
            body = dict(schema_version=1, key=key, boundary=boundary, parent=parent,
                        checkpoint=boundary in ('initial', 'committed'), files=entries,
                        directories=directories,
                        observed_at=datetime.now(timezone.utc).isoformat())
            identity = self._blob(_encoded(body))['object']
            result = dict(body, id=identity)
            _immutable(target, _encoded(result))
            return result

    def verify(self, identity):
        path = self.root/'objects'/_identity(identity)
        if path.is_symlink() or _hash(path.read_bytes()) != identity:
            raise ValueError('corrupt snapshot manifest')
        value = read(path)
        if value.get('schema_version') != 1 or value.get('boundary') not in BOUNDARIES:
            raise ValueError('unknown snapshot schema/boundary')
        if value['checkpoint'] != (value['boundary'] in ('initial', 'committed')):
            raise ValueError('invalid checkpoint marker')
        for name, item in value['files'].items():
            _member(name)
            blob = self.root/'objects'/_identity(item['object'])
            if blob.is_symlink() or fingerprint(blob) != dict(kind='file', sha256=item['object'], bytes=item['bytes']):
                raise ValueError('corrupt snapshot object')
        for name in value.get('directories', []):
            _member(name)
            if name in value['files'] or any(str(p) in value['files'] for p in PurePosixPath(name).parents):
                raise ValueError('snapshot directory collision')
        return dict(value, id=identity)

    def restore(self, identity, destination, *, view='researcher', evidence=False):
        """Materialize a fresh fork. No execution, path rewriting or version admission."""
        if view not in ('researcher', 'operator'):
            raise ValueError('unknown snapshot view')
        value = self.verify(identity)
        if not value['checkpoint'] and not evidence:
            raise ValueError('in-flight boundary is evidence, not a checkpoint')
        destination = Path(destination).absolute()
        if destination.exists() or destination.is_symlink():
            raise ValueError('restore requires a new directory')
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.snapshot-', dir=destination.parent))
        try:
            for name in value.get('directories', []):
                if view == 'operator' or name.split('/')[0] in ('changes', 'session'):
                    (staging/name).mkdir(parents=True, exist_ok=True)
            for name, item in value['files'].items():
                if view == 'researcher' and name.split('/')[0] not in ('changes', 'session'):
                    continue
                target = staging/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.root/'objects'/item['object'], target)
                target.chmod(0o700 if item['executable'] else 0o600)
            receipt = dict(schema_version=1, snapshot=identity, view=view,
                           mode='fork', checkpoint=value['checkpoint'], execution_resumed=False)
            (staging/'RESTORE.json').write_bytes(_encoded(receipt))
            # mkdir is exclusive, including for an existing empty destination.
            destination.mkdir(mode=0o700)
            staging.replace(destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return dict(status='completed', **receipt, destination=str(destination))

    def rollback(self, identity, store):
        """Select an admitted historical bundle only when live bytes still match."""
        from . import versions
        value = self.verify(identity)
        if not value['checkpoint']:
            raise ValueError('rollback requires a committed checkpoint')
        platform = _platform()
        captured = {k:v for k,v in value['files'].items() if k.startswith('platform/')}
        if set(platform) != set(captured) or any(
                fingerprint(path)['sha256'] != captured[name]['object'] for name, path in platform.items()):
            raise ValueError('platform changed; restore a separate fork and revalidate')
        entry = value['files']['audit/selection.json']
        selection = read(self.root/'objects'/entry['object'])
        return versions.activate(store, selection['active_version'])


def seal(workspace, store, key, boundary, state, *, candidates=(), session_files=None, journal=None, parent=None, research=None):
    """Controller boundary: platform + selected/proposed artifacts + private state.

    Session files are explicit researcher-visible inputs only. Native harness state
    stays external unless the caller deliberately exports a visible checkpoint.
    """
    from . import versions
    journal = journal or Journal(Path(workspace)/'.snapshots')
    event_path = journal.root/'events'/(key+'.json')
    if event_path.exists():
        prior = journal.event(key)
        if prior['boundary'] != boundary or prior['parent'] != parent:
            raise ValueError('snapshot boundary already bound')
        return prior
    files = _platform()
    store = Path(store)
    active = versions.current(store)
    selected = versions.load(store, active)
    identities = set(selected['members']) | set(candidates)
    bindings = {}
    expected_files, resources, directories = {}, {}, []
    for identity in sorted(identities):
        item = versions._item(store, identity)
        files[f'audit/candidates/{identity}.json'] = store/(identity+'.json')
        for source, expected in item['artifacts'].items():
            path = Path(source)
            if item['kind'] in ('weights', 'data'):
                resource = f'{identity}/{_hash(str(path).encode())}/{path.name}'
                resources[resource] = dict(identity=expected, requires_binding=True)
                bindings[source] = dict(resource=resource, identity=expected)
                continue
            prefix = 'changes/'+_hash(str(path).encode())+'/'+path.name
            bindings[source] = prefix
            if path.is_dir():
                directories.append(prefix)
                for child in sorted(path.rglob('*')):
                    if child.is_symlink():
                        raise ValueError('symlink in snapshot artifact')
                    if child.is_file():
                        name = prefix+'/'+child.relative_to(path).as_posix()
                        files[name] = child
                        expected_files[name] = fingerprint(child)['sha256']
                    elif child.is_dir():
                        directories.append(prefix+'/'+child.relative_to(path).as_posix())
            else:
                files[prefix] = path
                expected_files[prefix] = expected['sha256']
            if fingerprint(path) != expected:
                raise ValueError('candidate changed during snapshot')
    native = False
    session_files = dict(session_files or {})
    if research is not None:
        visible, private = research.files()
        session_files.update(visible)
        for name, path in private.items():
            files['audit/'+name] = path
        native = any(name.startswith('research/native/') and name.endswith('.jsonl') for name in private)
    for name, path in (session_files or {}).items():
        files['session/'+name] = path
    values = {'audit/controller.json': state,
              'audit/selection.json': dict(active_version=active, selected=selected, candidates=sorted(candidates)),
              'audit/bindings.json': bindings,
              'audit/context-coverage.json': dict(origin='controller',
                  visible_files=sorted(session_files or {}), native_session_exported=native,
                  process_state_restorable=False)}
    if resources:
        values['changes/resources.json'] = resources
    # Admission receipts remain operator-only and cannot be created by restore.
    for identity in {active, state.get('version')} - {None, 'empty'}:
        for suffix in (f'versions/{identity}.json', f'versions/results/{identity}.json'):
            if (store/suffix).is_file():
                files['audit/'+suffix] = store/suffix
    return journal.capture(key, boundary, files=files, values=values, parent=parent,
                           expected=expected_files, directories=directories)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list')
    for name in ('verify', 'restore', 'rollback'):
        command = sub.add_parser(name)
        command.add_argument('identity')
        if name == 'restore':
            command.add_argument('destination')
            command.add_argument('--view', choices=('researcher', 'operator'), default='researcher')
            command.add_argument('--evidence', action='store_true')
        if name == 'rollback':
            command.add_argument('--store', required=True)
    args = parser.parse_args(argv)
    journal = Journal(args.journal)
    if args.command == 'list':
        events = sorted((journal.event(p.stem) for p in (journal.root/'events').glob('*.json')),
                        key=lambda e: (e['observed_at'], e['key']))
        result = [dict(id=e['id'], key=e['key'], boundary=e['boundary'], checkpoint=e['checkpoint'],
                       parent=e['parent'], observed_at=e['observed_at'])
                  for e in events]
    elif args.command == 'verify':
        result = journal.verify(args.identity)
    elif args.command == 'rollback':
        result = journal.rollback(args.identity, args.store)
    else:
        result = journal.restore(args.identity, args.destination, view=args.view, evidence=args.evidence)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
