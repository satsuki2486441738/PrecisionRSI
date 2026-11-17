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
from .snapshots import Journal, seal
from .research import Research


def _infrastructure(workflow):
    """Classify construction harness failures, never native task/check failures."""
    workflow = workflow or {}
    action = (workflow.get('next_action') or
              next(iter(workflow.get('stages') or []), {}))
    execution = action.get('execution') or {}
    if execution.get('status') == 'launch_failed':
        return dict(kind='launch_failed')
    lines = (execution.get('logs') or {}).get('stdout', {}).get('tail', '').splitlines()
    stream = None
    if action.get('run'):
        try:
            stream = (Path(action['run'])/'stdout.bin').open(errors='replace')
        except OSError:
            pass
    failure = None
    try:
        for line in stream if stream is not None else lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get('type') != 'status' or event.get('phase') != 'turn_end':
                continue
            reason = event.get('reason')
            failure = None
            if isinstance(reason, dict) and reason.get('kind') == 'error':
                error = reason.get('error')
                code = error.get('code') if isinstance(error, dict) else None
                failure = dict(kind='harness_error', code=code if isinstance(code, str) else None)
    except OSError:
        pass
    finally:
        if stream is not None:
            stream.close()
    return failure


def _feedback(result, store):
    evaluation = (result.get('evaluation') or {}).get('evaluation') or {}
    measurement = evaluation.get('measurement') or {}
    measurements = {k:measurement[k] for k in ('construction','amortize_over_tasks') if k in measurement}
    for arm in ('baseline','candidate'):
        if arm not in measurement:
            continue
        value = measurement[arm]
        # Old receipts did not separate execution from amortized construction.
        # Keep those totals labelled; do not infer an unrecorded execution time.
        execution = value.get('execution')
        if execution is None:
            execution = dict(seconds=None if arm == 'candidate' and measurement.get('construction')
                             else value.get('seconds'), usage=value.get('usage'),
                             total_tokens=(value.get('usage') or {}).get('total_tokens'))
        measurements[arm] = dict({k:v for k,v in value.items()
            if k not in ('seconds','total_tokens','cost','usage','execution','including_construction')},
            execution=execution, including_construction=value.get('including_construction') or
                {k:value.get(k) for k in ('seconds','total_tokens','cost')})
    updates = [read(Path(store)/(identity+'.json')) for identity in result.get('candidates', [])]
    summary = dict(status=result.get('status'), version=result.get('version'),
        accepted=evaluation.get('accepted', False), no_update=result.get('no_update', False),
        errors=result.get('errors', []),
        selection=result.get('selection'),
        proposal_recovery=result.get('proposal_recovery'),
        construction=result.get('construction') or dict(seconds=result.get('spent_seconds'),usage=result.get('usage')),
        workflow=evaluation.get('workflow') or result.get('workflow'),
        measurements=measurements,
        updates=[dict({k:item[k] for k in ('id','kind','content','tags')},
                      artifacts=list(item['artifacts'])) for item in updates])
    infrastructure = _infrastructure(result.get('workflow'))
    if infrastructure:
        summary['infrastructure'] = infrastructure
    if result.get('evaluation') or any(e.get('stage')=='evaluation' for e in result.get('errors',[])):
        summary['evaluation'] = dict(seconds=(evaluation.get('workflow') or {}).get('spent_seconds'),
                                     usage=measurement.get('usage'))
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


def _stop(rounds, patience):
    if not rounds:
        return None
    last = rounds[-1]
    if last.get('feedback', {}).get('infrastructure'):
        return 'infrastructure failure'
    if last['result'].get('no_update'):
        return 'no useful update'
    if last['repeated']:
        return 'repeated candidate' if last['result'].get('candidates') else 'repeated failure'
    stalled = 0
    for prior in reversed(rounds):
        if prior['accepted']:
            break
        stalled += 1
    return 'no measured progress' if stalled >= patience else None


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
        research = Research(workspace/'research', dict(key=digest(binding)))
        journal = Journal(workspace/'.snapshots')
        # A boundary may have been sealed just before a controller-state write
        # was interrupted. Recover that commit without repeating agent work.
        for key in ('run-completed', f'round-{len(state["rounds"])+1:03d}-committed'):
            if (journal.root/'events'/(key+'.json')).exists():
                event = journal.event(key)
                recovered = read(journal.root/'objects'/event['files']['audit/controller.json']['object'])
                if recovered['key'] != state['key']:
                    raise ValueError('snapshot belongs to another evolution binding')
                state = dict(recovered, checkpoint=event['id'])
                write(path, state)
                if state.get('result'):
                    return state['result']
                break
        research.sync(state['rounds'])
        # A process crash consumes its reservation rather than granting free retries.
        if state.pop('reservation', None) is not None:
            state['reason'] = 'interrupted budget reservation consumed'
        remaining = cfg['budget_seconds']-state['seconds']
        deadline = time.monotonic()+remaining
        started = time.monotonic()
        spent = state['seconds']
        if not state.get('checkpoint'):
            initial = seal(workspace, store, 'checkpoint-000', 'initial',
                           dict(config=cfg, state=state), journal=journal, research=research)
            state['checkpoint'] = initial['id']
            write(path, state)
        reason = 'round limit'
        session_files = {}
        while len(state['rounds']) < cfg.get('max_rounds',8):
            active = research.private/'active.json'
            if active.exists():
                turn = read(active)['turn']
                receipt = research.private/('turn-'+turn+'.json')
                if receipt.exists() and read(receipt)['key'] == read(active)['key']:
                    active.unlink()
                else:
                    write(path, state)
                    return dict(status='needs_action', reason='unfinished researcher turn; reconcile explicitly',
                                turn=turn, state=str(path))
            stop = _stop(state['rounds'], cfg.get('patience',3))
            if stop:
                reason = stop
                break
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
            round_cfg['research'] = dict(root=str(research.root), binding=dict(key=digest(binding)),
                turn=f'round-{position+1:03d}')
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
            prefix = f'round-{position+1:03d}'
            before = seal(workspace, store, prefix+'-start', 'running',
                          dict(config=round_cfg, state=state), journal=journal,
                          parent=state['checkpoint'], research=research)
            try:
                result = improve(round_path,folder/'session',time_budget_seconds=allowance,
                    snapshot_context=dict(journal=journal, parent=before['id'], prefix=prefix, research=research))
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
            research.sync(state['rounds'])
            state.pop('pending',None)
            state.pop('reservation',None)
            state['seconds'] = spent+time.monotonic()-started
            submitted = journal.root/'events'/(prefix+'-submitted.json')
            session_files = {}
            flow = result.get('workflow') or {}
            receipt = flow.get('stages', [{}])[-1] if flow.get('status') == 'completed' else flow.get('next_action', {})
            for name, source in (('prompt.txt', Path(receipt['run'])/'prompt.txt' if receipt.get('run') else
                                 folder/'session/prompt-template.txt'),
                                 ('proposals.json', folder/'session/proposals.json')):
                if source.is_file():
                    session_files[name] = source
            parent_key = (prefix+'-submitted' if submitted.exists() else prefix+'-harness'
                          if (journal.root/'events'/(prefix+'-harness.json')).exists() else prefix+'-start')
            after = seal(workspace, store, prefix+'-committed', 'committed', state,
                         candidates=result.get('candidates', []), journal=journal,
                         parent=journal.event(parent_key)['id'], session_files=session_files, research=research)
            state['checkpoint'] = after['id']
            write(path, state)
        reason = _stop(state['rounds'], cfg.get('patience',3)) or reason
        state['seconds'] = spent+time.monotonic()-started
        construction_usage = usage_totals([r['result'] for r in state['rounds']])
        evaluation_usage = usage_totals([r['feedback']['evaluation'] for r in state['rounds']
                                        if 'evaluation' in r['feedback']])
        state['result'] = dict(status='needs_action' if reason == 'infrastructure failure' else 'completed',
            reason=reason,spent_seconds=state['seconds'],
            rounds=len(state['rounds']),accepted=sum(r['accepted'] for r in state['rounds']),
            usage=usage_totals([dict(usage=construction_usage),dict(usage=evaluation_usage)]),
            construction_usage=construction_usage,evaluation_usage=evaluation_usage,
            active_version=versions.current(store),state=str(path))
        terminal = seal(workspace, store, 'run-completed', 'committed', state,
                        candidates=state['rounds'][-1]['result'].get('candidates', []) if state['rounds'] else [],
                        journal=journal, parent=state['checkpoint'], session_files=session_files, research=research)
        state['checkpoint'] = terminal['id']
        write(path, state)
        return state['result']
