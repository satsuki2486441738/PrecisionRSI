"""run(CONFIG, workspace): budgeted external improvement with evaluation feedback.

CONFIG is an improvement config plus round_seconds (default 900), max_rounds
(default 8), patience (default 3). budget_seconds bounds the entire experiment.
Completed rounds replay without agent calls; an interrupted round is resumed.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import time

from .evidence import digest, usage_totals
from .improve import improve
from .memory import fingerprint, read, workspace_lock, write
from . import versions


def _feedback(result, store):
    evaluation = (result.get('evaluation') or {}).get('evaluation') or {}
    measurement = evaluation.get('measurement') or {}
    updates = [read(Path(store)/(identity+'.json')) for identity in result.get('candidates', [])]
    summary = dict(status=result.get('status'), version=result.get('version'),
        accepted=evaluation.get('accepted', False), no_update=result.get('no_update', False),
        errors=result.get('errors', []),
        proposal_recovery=result.get('proposal_recovery'),
        construction=dict(seconds=result.get('spent_seconds'),usage=result.get('usage')),
        workflow=evaluation.get('workflow') or result.get('workflow'),
        measurements={k:measurement[k] for k in
            ('baseline','candidate','construction','amortize_over_tasks') if k in measurement},
        updates=[{k:item[k] for k in ('id','kind','content','tags','artifacts')} for item in updates])
    # Paths identify retained details; full workflow transcripts stay outside prompts.
    if summary['workflow']:
        flow = summary['workflow']
        summary['workflow'] = {k:flow[k] for k in ('status','reason','state','next_action') if k in flow}
        if flow.get('next_action'):
            action = flow['next_action']
            compact = {k:action[k] for k in ('stage','skill','reason') if k in action}
            execution = action.get('execution',{})
            if execution:
                compact['execution'] = {k:execution[k] for k in ('status','returncode') if k in execution}
                logs = execution.get('logs',{})
                # Keep native diagnostics, not model reasoning/tool events embedded in stdout.
                for line in logs.get('stdout',{}).get('tail','').splitlines():
                    try:
                        value = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(value,dict) and ('reason' in value or 'error' in value):
                        compact['diagnostic'] = {k:value[k] for k in ('status','reason','error','selection') if k in value}
                if 'diagnostic' not in compact and logs.get('stderr',{}).get('tail'):
                    compact['error'] = logs['stderr']['tail'][-1200:]
            summary['workflow']['next_action'] = compact
    signature = digest([dict(kind=i['kind'], content=i['content'], tags=i['tags'],
        artifacts=sorted(digest(fingerprint(p)) for p in i['artifacts'])) for i in updates]
        if updates else summary)
    return summary, signature


def run(config, workspace):
    from .loop import _positive
    config = Path(config).resolve()
    cfg = read(config)
    if cfg.get('schema_version') != 1 or not _positive(cfg.get('budget_seconds')):
        raise ValueError('evolution requires schema_version=1 and positive budget_seconds')
    for key, default in (('round_seconds',900), ('max_rounds',8), ('patience',3)):
        value = cfg.get(key, default)
        if not _positive(value) or (key != 'round_seconds' and type(value) is not int):
            raise ValueError('invalid evolution '+key)
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    store = (config.parent/cfg['store']).resolve()
    binding = dict(config=cfg, runtime=versions._runtime())
    with workspace_lock(workspace):
        path = workspace/'evolution.json'
        state = read(path) if path.exists() else dict(key=digest(binding), rounds=[], seconds=0)
        if state['key'] != digest(binding):
            raise ValueError('evolution config or runtime changed; use a new workspace')
        if state.get('result'):
            return state['result']
        # A process crash consumes its reservation rather than granting free retries.
        if state.pop('reservation', None) is not None:
            state['reason'] = 'interrupted budget reservation consumed'
        remaining = cfg['budget_seconds']-state['seconds']
        deadline = time.monotonic()+remaining
        started = time.monotonic()
        spent = state['seconds']
        reason = 'round limit'
        while len(state['rounds']) < cfg.get('max_rounds',8):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                reason = 'budget exhausted'
                break
            position = len(state['rounds'])
            if not state.get('pending'):
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                state['pending'] = str(workspace/f'compare-improve{position+1}-{stamp}')
            folder = Path(state['pending'])
            folder.mkdir(parents=True,exist_ok=True)
            round_cfg = {k:v for k,v in cfg.items() if k not in ('round_seconds','max_rounds','patience')}
            round_cfg['store'] = str(store)
            for key in ('evaluation_plan',):
                if cfg.get(key):
                    round_cfg[key] = str((config.parent/cfg[key]).resolve())
            round_cfg['evidence'] = []
            for source in cfg.get('evidence',[]):
                value = dict(source) if isinstance(source,dict) else dict(state=source)
                value['state'] = str((config.parent/value['state']).resolve())
                round_cfg['evidence'].append(value)
            round_cfg['feedback'] = cfg.get('feedback',[]) + [r['feedback'] for r in state['rounds'][-3:]]
            allowance = min(cfg.get('round_seconds',900), remaining)
            round_path = folder/'improve.json'
            if not round_path.exists():
                round_cfg['budget_seconds'] = allowance
                write(round_path,round_cfg)
            state['reservation'] = allowance
            state['seconds'] = spent+time.monotonic()-started+allowance
            write(path,state)
            try:
                result = improve(round_path,folder/'session',time_budget_seconds=allowance)
                feedback, signature = _feedback(result,store)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result = dict(status='needs_action',reason=str(exc))
                feedback = dict(status='needs_action',errors=[dict(reason=str(exc))])
                signature = digest(feedback)
            accepted = feedback.get('accepted',False)
            repeated = signature in {r['signature'] for r in state['rounds']}
            entry = dict(workspace=str(folder),result=result,feedback=feedback,signature=signature,
                accepted=accepted,repeated=repeated)
            state['rounds'].append(entry)
            state.pop('pending',None)
            state.pop('reservation',None)
            state['seconds'] = spent+time.monotonic()-started
            write(path,state)
            stalled = 0
            for prior in reversed(state['rounds']):
                if prior['accepted']:
                    break
                stalled += 1
            if result.get('no_update') or repeated or stalled >= cfg.get('patience',3):
                reason = ('no useful update' if result.get('no_update') else
                    'repeated candidate' if repeated and result.get('candidates') else
                    'repeated failure' if repeated else 'no measured progress')
                break
        state['seconds'] = spent+time.monotonic()-started
        state['result'] = dict(status='completed',reason=reason,spent_seconds=state['seconds'],
            rounds=len(state['rounds']),accepted=sum(r['accepted'] for r in state['rounds']),
            usage=usage_totals([r['result'] for r in state['rounds']]),
            active_version=versions.current(store),state=str(path))
        write(path,state)
        return state['result']
