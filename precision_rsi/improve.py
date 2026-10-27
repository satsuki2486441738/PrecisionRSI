"""improve(CONFIG, workspace, sources=None, time_budget_seconds=None) -> receipt.

Config: schema_version=1, store, evidence:[loop-state path or {state,family}],
agent:{argv,timeout,usage_format?,execution?}, budget_seconds, max_updates?:4,
evaluation_plan?:version-evaluator template, held_out_families?:[], splits?:{}.
Paths are relative to CONFIG. One bounded independent harness invocation writes
{updates:[]} to the indicated proposals.json. Native receipts prevent replay calls.
"""
import json
from pathlib import Path
import sys
import time

from .dataset import build as build_dataset
from .evidence import brief, collect, digest, verify
from .harness import price, usage
from .memory import propose, read, run_workflow, select, workspace_lock, write
from . import versions


def _stable(path, value):
    if path.exists() and read(path) != value:
        raise ValueError('improvement input changed; use a new workspace')
    if not path.exists():
        write(path, value)


def _sources(config, spec):
    result = []
    for value in spec.get('evidence', []):
        value = dict(value) if isinstance(value, dict) else dict(state=value)
        value['state'] = str((config.parent/value['state']).resolve())
        result.append(value)
    return result


def improve(config, workspace, sources=None, time_budget_seconds=None):
    from .loop import _positive, _render
    config = Path(config).resolve()
    cfg = read(config)
    if cfg.get('schema_version') != 1 or not _positive(cfg.get('budget_seconds')):
        raise ValueError('improvement needs schema_version=1 and positive budget_seconds')
    if time_budget_seconds is not None and not _positive(time_budget_seconds):
        raise ValueError('positive outer improvement budget required')
    maximum = cfg.get('max_updates', 4)
    if type(maximum) is not int or maximum <= 0:
        raise ValueError('positive max_updates required')
    index = collect(_sources(config, cfg) + list(sources or []))
    if not index['tasks']:
        raise ValueError('improvement needs task evidence')
    if any(t['family'] in cfg.get('held_out_families', []) for t in index['tasks']):
        raise ValueError('improvement evidence overlaps held-out task families')
    if cfg.get('evaluation_plan') and any(not t['family'] for t in index['tasks']):
        raise ValueError('declare source task families before independent evaluation')
    store = (config.parent/cfg['store']).resolve()
    store.mkdir(parents=True, exist_ok=True)
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    with workspace_lock(workspace):
        state_path = workspace/'improvement.json'
        state = read(state_path) if state_path.exists() else dict(base_version=versions.current(store), history=[])
        base = state['base_version']
        runtime = versions.load(store, base)['runtime']
        _stable(workspace/'evidence.json', index)
        _stable(workspace/'config.json', {k:v for k,v in cfg.items() if k != 'budget_seconds'})
        remaining = cfg['budget_seconds'] - sum(a['seconds'] for a in state['history'])
        if time_budget_seconds is not None:
            remaining = min(remaining, time_budget_seconds)
        if remaining <= 0:
            return dict(status='needs_action', reason='improvement budget exhausted', state=str(state_path))
        attempt = dict(seconds=remaining, status='running')
        state['history'].append(attempt)
        write(state_path, state)
        started = time.monotonic()
        deadline = started + remaining
        try:
            agent = cfg['agent']
            if not _positive(agent.get('timeout')) or agent.get('usage_format') not in (None, 'dsh-jsonl'):
                raise ValueError('positive agent timeout and supported usage format required')
            _stable(workspace/'agent.json', agent)
            tags = {tag for t in index['tasks'] for tag in t['tags']}
            interfaces = select(store,tags,limit_chars=2000,version=base)
            prompt = ('Produce one small reusable improvement justified by the task evidence. '
                'Use the supplied observations and complete interface directly. '
                'Do not read host implementation source or full harness transcripts. '
                'Read an individual evidence artifact only to resolve a specific missing fact. '
                'Build the minimum candidate, '
                'save proposals and return. Build new files under the artifact '
                'directory and return {"updates":[]} when no useful update is justified. '
                'Each update has kind, content (short interface/rule), tags, artifacts (absolute persistent '
                'file paths), and benefit. Kinds: tool, context, procedure, data. '
                'context stores a concise rule; tool stores callable code plus a test. '
                'procedure declares entrypoint: an artifact JSON with instruction and/or '
                'skill_paths mapping framework-adapt/precision-compare/precision-repair to declared SKILL.md artifacts. '
                'Data artifacts are JSONL: {source: indexed task ID, messages: [{role,content}]}; '
                'labels refer to whole-task outcomes. Use replaces:[candidate IDs] to replace base members. '
                'One limitation may need a coupled set of updates, up to max_updates. '
                'The host builds datasets, evaluates and publishes; benefits are hypotheses until measured.\n'
                + json.dumps(dict(base_version=base, evidence=brief(index)[:cfg.get('summary_limit',12)],
                    available_interfaces=interfaces,
                    feedback=cfg.get('feedback', []),
                    tool_help=[sys.executable, str(Path(__file__).resolve().parents[1]/
                        'skills/precision-compare/scripts/tool_help.py')],
                    evidence_count=len(index['tasks']),
                    evidence_index=str(workspace/'evidence.json'), output='{run}/proposals.json',
                    artifacts='{run}/artifacts', max_updates=maximum), ensure_ascii=False))
            prompt_path = workspace/'prompt-template.txt'
            if prompt_path.exists() and prompt_path.read_text() != prompt:
                raise ValueError('improvement prompt changed; use a new workspace')
            if not prompt_path.exists():
                prompt_path.write_text(prompt)
            stage = dict(id='improve', skill='precision-repair', timeout=agent['timeout'],
                argv=[sys.executable, '-B', str(Path(__file__).with_name('harness.py')),
                      str(workspace/'agent.json'), str(prompt_path), '{run}'],
                inputs=[str(workspace/'config.json'), str(workspace/'evidence.json'), str(workspace/'agent.json'),
                        str(prompt_path), str(Path(__file__).with_name('harness.py'))],
                outputs={'proposals':'proposals.json', 'artifacts':'artifacts'})
            if agent.get('execution'):
                stage['execution'] = str((config.parent/agent['execution']).resolve())
            plan = dict(schema_version=1, id='system-improvement', root=str(workspace),
                        budget_seconds=cfg['budget_seconds'], stages=[stage])
            plan_path = workspace/'agent-plan.json'
            if not plan_path.exists() or read(plan_path) != plan:
                write(plan_path, plan)
            workflow = run_workflow(plan_path, workspace/'agent', time_budget_seconds=remaining)
            result = dict(status='needs_action', workflow=workflow, candidates=[], datasets=[],
                          errors=[], version=None, evaluation=None, state=str(state_path))
            receipt = workflow['stages'][0] if workflow['status']=='completed' else workflow.get('next_action', {})
            stdout = Path(receipt['run'])/'stdout.bin' if receipt.get('run') else None
            result['usage'] = usage(stdout, agent.get('usage_format')) if stdout and stdout.is_file() else None
            if workflow['status'] == 'completed':
                for task in index['tasks']:
                    verify(task)
                if versions._runtime() != runtime:
                    raise ValueError('improvement changed the running system')
                updates = read(receipt['outputs']['proposals']['path'])['updates']
                if not isinstance(updates, list) or len(updates) > maximum:
                    raise ValueError('updates must be a bounded list')
                runtime_ids = []
                for position, update in enumerate(updates):
                    try:
                        update = dict(update)
                        if update.get('kind') == 'data':
                            dataset = build_dataset(update['artifacts'], index, workspace/'datasets',
                                cfg.get('held_out_families', []), cfg.get('splits'))
                            update['artifacts'] = dataset['artifacts']
                            result['datasets'].append(dataset)
                        update['base_version'] = base
                        identity = propose(store, update, dict(evidence=index['id'],
                            families=sorted({t['family'] for t in index['tasks'] if t['family']}),
                            improvement=str(state_path)))
                        result['candidates'].append(identity)
                        if update['kind'] == 'data':
                            with workspace_lock(store):
                                candidate_path = store/(identity+'.json')
                                candidate = read(candidate_path)
                                candidate['status'] = 'ready'
                                write(candidate_path, candidate)
                        if update['kind'] in ('tool', 'context', 'procedure'):
                            runtime_ids.append(identity)
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        result['errors'].append(dict(update=position, reason=str(exc)))
                if runtime_ids:
                    version = versions.prepare(store, runtime_ids, base)
                    result['version'] = version['id']
                    if cfg.get('evaluation_plan') and deadline > time.monotonic():
                        construction = dict(seconds=sum(a['seconds'] for a in state['history'][:-1])+
                                            time.monotonic()-started, usage=result['usage'],
                                            cost=price(result['usage'],cfg.get('pricing')))
                        if not (workspace/'construction.json').exists():
                            write(workspace/'construction.json', construction)
                        template = (config.parent/cfg['evaluation_plan']).resolve()
                        evaluation = _render(read(template),
                            {'{version_id}':version['id'], '{baseline_version}':base,
                             '{store}':str(store), '{construction_file}':str(workspace/'construction.json'),
                             '{round_workspace}':str(workspace)})
                        evaluation['root'] = str((template.parent/evaluation.get('root', '.')).resolve())
                        evaluation_path = workspace/'evaluation-plan.json'
                        write(evaluation_path, evaluation)
                        try:
                            result['evaluation'] = versions.evaluate(store, version['id'], evaluation_path,
                                workspace/'evaluation', time_budget_seconds=deadline-time.monotonic())
                        except (OSError, ValueError, KeyError, TypeError) as exc:
                            result['errors'].append(dict(stage='evaluation', reason=str(exc)))
                result['status'] = 'completed' if not result['errors'] else 'needs_action'
                result['no_update'] = not updates
            state['result'] = result
        finally:
            attempt.update(seconds=time.monotonic()-started, status='finished')
            write(state_path, state)
        result['spent_seconds'] = sum(a['seconds'] for a in state['history'])
        return result


def maybe(config, sources, series_state, workspace, remaining):
    """Trigger on new evidence, reusing a recorded round on series replay."""
    from datetime import datetime, timezone
    config = Path(config).resolve()
    cfg = read(config)
    index = collect(_sources(config,cfg) + list(sources))
    rounds = series_state.setdefault('improvements', [])
    identity = digest(dict(config={k:v for k,v in cfg.items() if k!='budget_seconds'}, evidence=index['id']))
    prior = next((r for r in rounds if r['key']==identity), None)
    if prior is not None:
        if prior.get('result', {}).get('status') == 'completed':
            return prior['result']
    else:
        seen = {identity for r in rounds for identity in r['sources']}
        new = [t for t in index['tasks'] if t['id'] not in seen]
        trigger = cfg.get('trigger', {})
        minimum = trigger.get('min_new_tasks', 2)
        if type(minimum) is not int or minimum <= 0:
            raise ValueError('positive min_new_tasks required')
        if len(new) < minimum and not (trigger.get('on_failure', True) and
                                      any(t['status']=='needs_action' for t in new)):
            return None
        stamp = datetime.now(timezone.utc).strftime('compare-improve'+str(len(rounds)+1)+'-%Y%m%dT%H%M%SZ')
        prior = dict(key=identity, sources=[t['id'] for t in index['tasks']],
                     workspace=str(Path(workspace)/'improvements'/stamp))
        rounds.append(prior)
        write(Path(workspace)/'series.json', series_state)
    try:
        prior['result'] = improve(config, prior['workspace'], sources, remaining)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        prior['result'] = dict(status='needs_action', reason=str(exc))
    return prior['result']
