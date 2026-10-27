"""Immutable runtime bundles: prepare, evaluate, activate, current and members.

A version binds tools, context and procedure overlays, their artifact identities,
its parent and runner/skill source identities. Evaluation plans and measurements
bind version_id and baseline_version. Only evaluated versions can be activated.
"""
from pathlib import Path

from .evidence import digest
from .memory import _entry, _unchanged, fingerprint, improvement, measure, read, workspace_lock, write


def _runtime():
    root = Path(__file__).resolve().parents[1]
    files = list((root/'precision_rsi').glob('*.py'))
    files += list((root/'skills').glob('*/SKILL.md'))
    files += list((root/'skills/precision-compare/scripts').glob('*.py'))
    return {str(p.relative_to(root)):fingerprint(p) for p in sorted(files)}


def current(store):
    path = Path(store)/'versions/current.json'
    return read(path)['id'] if path.exists() else 'empty'


def _item(store, identity):
    item = read(_entry(store, identity))
    core = {k:v for k,v in item.items() if k not in ('id', 'status', 'evaluations')}
    if digest(core) != identity or not _unchanged(item):
        raise ValueError('versioned candidate or artifact changed')
    return item


def load(store, identity):
    if identity == 'empty':
        return dict(id='empty', members=[], runtime=_runtime())
    path = Path(store)/'versions'/(_entry(store, identity).name)
    value = read(path)
    if digest({k:v for k,v in value.items() if k != 'id'}) != identity or value['runtime'] != _runtime():
        raise ValueError('system version or runtime changed; evaluate a new version')
    for member in value['members']:
        _item(store, member)
    return value


def members(store, identity):
    return [_item(store, member) for member in load(store, identity)['members']]


def prepare(store, identities, base_version=None):
    store = Path(store)
    base_version = current(store) if base_version is None else base_version
    inherited = set(load(store, base_version)['members'])
    additions, replacements = set(), set()
    for identity in identities:
        item = _item(store, identity)
        if item['kind'] not in ('tool', 'context', 'procedure'):
            raise ValueError('runtime versions contain tools, context and procedures')
        if item.get('base_version', base_version) != base_version:
            raise ValueError('candidate was built for another base version')
        replacements.update(item.get('replaces', []))
        additions.add(identity)
    if not replacements <= inherited:
        raise ValueError('replacement is absent from base version')
    selected = sorted((inherited-replacements)|additions)
    families = {family for identity in selected for family in
                _item(store,identity).get('source', {}).get('families', [])}
    value = dict(schema_version=1, base_version=base_version,
                 members=selected, evidence_families=sorted(families), runtime=_runtime())
    value['id'] = digest(value)
    folder = store/'versions'
    folder.mkdir(parents=True, exist_ok=True)
    with workspace_lock(store):
        path = folder/(value['id']+'.json')
        if path.exists() and read(path) != value:
            raise ValueError('immutable version changed')
        if not path.exists():
            write(path, value)
    return value


def activate(store, identity, expected_current=None):
    """Enable an admitted version, restore one, or return to empty memory."""
    load(store, identity)
    if identity != 'empty':
        result = read(Path(store)/'versions/results'/(identity+'.json'))
        if not result['accepted']:
            raise ValueError('version has no passing evaluation')
    (Path(store)/'versions').mkdir(parents=True, exist_ok=True)
    with workspace_lock(Path(store)):
        old = current(store)
        if expected_current is not None and old not in (expected_current, identity):
            raise ValueError('active base advanced during evaluation')
        if old != identity:
            write(Path(store)/'versions/current.json', dict(id=identity, previous=old))
    return dict(status='active', id=identity, previous=old)


def evaluate(store, identity, plan, workspace, time_budget_seconds=None):
    version = load(store, identity)
    binding = dict(version_id=identity, baseline_version=version['base_version'])
    workflow, measurement = measure(plan, workspace, binding, time_budget_seconds)
    load(store, identity)
    if measurement is not None and version.get('evidence_families'):
        families = measurement.get('families')
        if not isinstance(families, list) or not families or set(families) & set(version['evidence_families']):
            raise ValueError('evaluation must use independent task families')
    accepted = measurement is not None and improvement(measurement)
    record = dict(id=identity, accepted=accepted, workflow=workflow, measurement=measurement)
    folder = Path(store)/'versions/results'
    folder.mkdir(parents=True, exist_ok=True)
    with workspace_lock(Path(store)):
        path = folder/(identity+'.json')
        prior = read(path) if path.exists() else dict(history=[])
        history = prior['history']
        if not history or history[-1]['measurement'] != measurement:
            history.append(record)
        write(path, dict(accepted=accepted, history=history))
    result = dict(id=identity, status='candidate', evaluation=record)
    if accepted:
        result.update(activate(store, identity, expected_current=version['base_version']))
    return result
