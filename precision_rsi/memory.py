"""Bounded, condition-matched experience; candidates activate through held-out evaluation.

propose(store, proposal, source) retains a candidate without changing active memory.
evaluate(store, candidate_id, plan, workspace, time_budget_seconds=None) executes a
task-owned evaluator through workflow.run_workflow. Its final output named
``evaluation`` contains candidate_id, mode, scope, baseline and candidate.
Closed measurements: correct:bool, seconds:positive float, optional cost:float.
Open measurements additionally provide budget and quality (larger is better).
An evaluator plan declares workload.purpose='held_out_evaluation' and candidate_id.
select(store, tags, limit_chars) returns active, unchanged tool/context interfaces.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/precision-compare/scripts'))

from workflow import fingerprint, run_workflow, workspace_lock


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def propose(store, proposal, source):
    """Retain verified-task experience as a candidate; artifact paths must be absolute."""
    if proposal.get('kind') not in ('tool', 'context', 'procedure', 'data', 'weights'):
        raise ValueError('unknown experience kind')
    if not isinstance(proposal.get('content'), str) or not proposal['content'].strip():
        raise ValueError('experience content required')
    tags = proposal.get('tags', [])
    if not isinstance(tags, list) or not all(isinstance(t, str) and t for t in tags):
        raise ValueError('experience tags must be strings')
    artifacts = proposal.get('artifacts', [])
    if not isinstance(artifacts, list) or not all(isinstance(p, str) and Path(p).is_absolute() for p in artifacts):
        raise ValueError('artifact paths must be absolute')
    if proposal['kind'] in ('tool', 'procedure', 'data', 'weights') and not artifacts:
        raise ValueError(proposal['kind'] + ' experience needs a versioned artifact')
    item = dict(kind=proposal['kind'], content=proposal['content'], tags=sorted(set(tags)),
                artifacts={p: fingerprint(p) for p in artifacts}, source=source)
    for key in ('entrypoint', 'benefit', 'base_version', 'replaces'):
        if key in proposal:
            item[key] = proposal[key]
    if 'replaces' in item and (not isinstance(item['replaces'], list) or
            not all(isinstance(v, str) and len(v) == 64 for v in item['replaces'])):
        raise ValueError('replaces must list candidate IDs')
    if item['kind'] == 'procedure':
        procedure(item)
    identity = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    path = store / (identity + '.json')
    with workspace_lock(store):
        if not path.exists():
            write(path, dict(item, id=identity, status='candidate', evaluations=[]))
    return identity


def _entry(store, candidate_id):
    if len(candidate_id) != 64 or any(c not in '0123456789abcdef' for c in candidate_id):
        raise ValueError('invalid candidate ID')
    return Path(store) / (candidate_id + '.json')


def _unchanged(item):
    try:
        return all(fingerprint(p) == identity for p, identity in item['artifacts'].items())
    except (OSError, ValueError):
        return False


def procedure(item):
    """Versioned instruction/skill-route overlay; all referenced files are identified."""
    entrypoint = item.get('entrypoint')
    if entrypoint not in item['artifacts']:
        raise ValueError('procedure entrypoint must be a declared artifact')
    value = read(entrypoint)
    if not isinstance(value, dict) or set(value) - {'instruction', 'skill_paths'}:
        raise ValueError('procedure supports instruction and skill_paths')
    if 'instruction' in value and (not isinstance(value['instruction'], str) or not value['instruction'].strip()):
        raise ValueError('procedure instruction must be nonempty')
    routes = value.get('skill_paths', {})
    if not isinstance(routes, dict) or any(k not in ('framework-adapt', 'precision-compare', 'precision-repair')
            or p not in item['artifacts'] for k, p in routes.items()):
        raise ValueError('procedure skill paths must be declared skill artifacts')
    if not value.get('instruction') and not routes:
        raise ValueError('procedure must provide an instruction or a skill route')
    return value


def select(store, tags=(), limit_chars=4000, version=None):
    """Load only active, applicable, unchanged interfaces within a character budget."""
    if type(limit_chars) is not int or limit_chars < 0:
        raise ValueError('nonnegative memory character budget required')
    from .versions import current, members
    version = (None if version == 'legacy' else version or
               (current(store) if (Path(store)/'versions/current.json').exists() else None))
    items = members(store, version) if version else [read(p) for p in sorted(Path(store).glob('*.json'))]
    result, used = [], 0
    for item in items:
        if ((version is None and item['status'] != 'active') or
                item['kind'] not in ('tool', 'context', 'procedure') or not set(item['tags']) <= set(tags)
                or not _unchanged(item)):
            continue
        value = dict(id=item['id'], kind=item['kind'], content=item['content'],
                     artifacts=list(item['artifacts']))
        if item['kind'] == 'procedure':
            value['procedure'] = procedure(item)
        size = len(json.dumps(value, ensure_ascii=False))
        if used + size <= limit_chars:
            result.append(value)
            used += size
    return result


def _number(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0 and (not positive or value > 0)


def improvement(result):
    """Task-owned correctness first; closed cost/time dominance or open quality gain."""
    a, b = result['baseline'], result['candidate']
    if not isinstance(result.get('scope'), list) or not result['scope']:
        raise ValueError('evaluation scope required')
    for value in (a, b):
        if type(value.get('correct')) is not bool or not _number(value.get('seconds'), True):
            raise ValueError('correctness and measured seconds required')
    if result['mode'] not in ('closed', 'open'):
        raise ValueError('evaluation mode must be closed or open')
    if not b['correct']:
        return False
    if result['mode'] == 'open':
        for value in (a, b):
            if not _number(value.get('budget'), True) or not _number(value.get('quality')):
                raise ValueError('open evaluation requires equal budgets and nonnegative quality')
        return a['budget'] == b['budget'] and (not a['correct'] or b['quality'] > a['quality'])
    if not a['correct']:
        return True
    metrics = ['seconds']
    for metric in ('cost', 'total_tokens'):
        if metric in a or metric in b:
            if not all(_number(x.get(metric)) for x in (a, b)):
                raise ValueError('both costs required when evaluating cost' if metric=='cost'
                                 else 'both token totals required when evaluating usage')
            metrics.append(metric)
    return all(b[k] <= a[k] for k in metrics) and any(b[k] < a[k] for k in metrics)


def evaluate(store, candidate_id, plan, workspace, time_budget_seconds=None):
    """Activate a tool/context candidate only after a declared held-out comparison."""
    path = _entry(store, candidate_id)
    item = read(path)
    if item['kind'] not in ('tool', 'context'):
        raise ValueError('data and weights are retained for future training adapters')
    if not _unchanged(item):
        raise ValueError('candidate artifact changed; propose the new version')
    result, measurement = measure(plan, workspace, {'candidate_id':candidate_id}, time_budget_seconds)
    accepted = measurement is not None and improvement(measurement)
    if not _unchanged(item):
        raise ValueError('candidate artifact changed during evaluation')
    record = dict(accepted=accepted, workflow=result, measurement=measurement)
    with workspace_lock(Path(store)):
        item = read(path)
        if not item['evaluations'] or item['evaluations'][-1]['measurement'] != measurement:
            item['evaluations'].append(record)
        # A later failed evaluation removes an earlier admission.
        item['status'] = 'active' if accepted else 'candidate'
        write(path, item)
    return dict(id=candidate_id, status=item['status'], evaluation=record)


def measure(plan, workspace, binding, time_budget_seconds=None):
    """Shared native evaluation interface, bound to the exact candidate/version."""
    workload = read(plan).get('workload', {})
    if workload.get('purpose') != 'held_out_evaluation' or any(workload.get(k) != v for k,v in binding.items()):
        raise ValueError('evaluation must bind its subject and declare held-out scope')
    result = run_workflow(plan, workspace, time_budget_seconds=time_budget_seconds)
    measurement = None
    if result['status'] == 'completed':
        output = result['stages'][-1]['outputs'].get('evaluation')
        if output is None:
            raise ValueError('final evaluator output must be named evaluation')
        measurement = read(output['path'])
        if any(measurement.get(k) != v for k,v in binding.items()):
            raise ValueError('measurement belongs to another candidate/version')
    return result, measurement
