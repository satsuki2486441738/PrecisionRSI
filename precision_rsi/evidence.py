"""collect(sources) -> bounded task index; verify(task) checks retained evidence.

Sources are loop.json paths or {state: path, family: task-family} declarations.
Both completed and incomplete tasks are indexed. Families are declared by the
task owner. Raw harness output and native artifacts stay outside summaries.
"""
import hashlib
import json
import math
from pathlib import Path

from .memory import fingerprint, read
from .harness import activity


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def usage_totals(attempts):
    """Sum complete measurements only; any missing observation remains unknown."""
    usages = [a.get('usage') for a in attempts]
    return {key: (sum(u[key] for u in usages) if all(isinstance(u, dict) and
            type(u.get(key)) is int and u[key] >= 0 for u in usages) else None)
            for key in ('input_tokens', 'output_tokens', 'total_tokens', 'cached_input_tokens')}


def _observation(path):
    """Expose existing report scalars so diagnosis does not require raw-log replay."""
    path = Path(path)
    if path.suffix != '.json' or path.stat().st_size > 65536:
        return None
    try:
        value = read(path)
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    scalars = {k:v for k,v in value.items() if v is None or type(v) in (bool,int,float,str)}
    return {k:(v[:240] if isinstance(v,str) else str(v) if isinstance(v,float) and not math.isfinite(v)
               else v) for k,v in list(scalars.items())[:16]}


def collect(sources):
    tasks = {}
    for source in sources:
        declaration = source if isinstance(source, dict) else dict(state=str(source))
        path = Path(declaration['state']).resolve()
        if path.is_dir():
            path /= 'loop.json'
        state = read(path)
        result = state.get('result')
        if not isinstance(result, dict) or result.get('status') not in ('completed', 'needs_action'):
            raise ValueError('evidence requires a finished task attempt: ' + str(path))
        cfg = read(state['task'])
        family = declaration.get('family', cfg.get('family'))
        if family is not None and (not isinstance(family, str) or not family.strip()):
            raise ValueError('task family must be a nonempty string')
        agents = [a for a in state['history'] if a['kind'] == 'agent']
        native_path = path.parent / 'native/state.json'
        native = read(native_path) if native_path.is_file() else dict(history=[])
        refs, stages, observations = {}, [], []
        for attempt in native['history']:
            outputs = {name: value['path'] for name, value in attempt.get('outputs', {}).items()}
            for name in outputs.values():
                refs[name] = fingerprint(name)
            stages.append(dict(stage=attempt['stage'], status=attempt['status'],
                               reason=attempt.get('reason'), run=attempt.get('run'), outputs=outputs))
            for name, filename in outputs.items():
                value = _observation(filename)
                if value:
                    observations.append(dict(stage=attempt['stage'], status=attempt['status'],
                                             output=name, values=value, path=filename))
        actions = []
        for attempt in agents:
            if attempt.get('raw_output'):
                refs[attempt['raw_output']] = fingerprint(attempt['raw_output'])
                summary = activity(attempt['raw_output'], cfg['agent'].get('usage_format'))
                if summary:
                    actions.append(summary)
        workflow = result.get('workflow') or {}
        item = dict(task=state['task'], task_key=state['key'], state=str(path), family=family,
                    tags=cfg.get('memory', {}).get('tags', []),
                    goal=cfg['goal'][:600], workload=workflow.get('workload'),
                    status=result['status'], reason=result.get('reason'), stages=stages,
                    agent_calls=len(agents), usage=usage_totals(agents),
                    agent_activity=actions[-4:],
                    observations=observations[-12:],
                    seconds=sum(a.get('wall_seconds', 0) for a in native['history']) +
                            sum(a['wall_seconds'] for a in agents),
                    system_version=state.get('system_version'), references=refs)
        # Replaying receipts does not create another experience for the same work.
        item['id'] = digest({k:v for k,v in item.items() if k != 'seconds'})
        tasks[item['id']] = item
    values = sorted(tasks.values(), key=lambda t:t['id'])
    return dict(schema_version=1, tasks=values, id=digest(values))


def verify(task):
    if task['id'] != digest({k:v for k,v in task.items() if k not in ('id', 'seconds')}):
        raise ValueError('task evidence summary changed')
    if any(fingerprint(path) != identity for path, identity in task['references'].items()):
        raise ValueError('task evidence artifact changed')


def brief(index):
    """Prompt view: results and costs, with paths for optional evidence inspection."""
    return [{k:v for k,v in task.items() if k not in ('references', 'stages')}
            for task in index['tasks']]
