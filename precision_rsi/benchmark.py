"""run(SUITE, workspace, version_id, baseline_version, seconds) -> measurement.

Suite: schema_version=1, store, cases:[{id,family,baseline,candidate}],
construction?:JSON receipt, amortize_over_tasks?:positive integer.
Cases declare independent workload copies. Both arms use the same agent/runtime,
task semantics and initial source bytes. Each arm runs once; replay retains its
measured result. The host supplies disjoint improvement/evaluation task families.
"""
import json
from pathlib import Path
import time

from .evidence import collect, digest, usage_totals, verify
from .memory import fingerprint, read, select, workspace_lock, write
from .harness import price
from . import versions


def _signature(task):
    from .loop import _config, _render
    path, cfg, plan, root, protected, key = _config(task)
    spec = read(plan)
    normalized = _render(spec, {str(root):'{workload}'})
    normalized.pop('root', None)
    sources = [[fingerprint(root/name) if (root/name).exists() else {'kind':'missing'}
                for name in s.get('inputs', [])] for s in spec['stages']]
    memory = cfg.get('memory', {})
    return dict(goal=cfg['goal'], interface=cfg.get('interface'), plan=normalized, sources=sources, agent=cfg['agent'],
                retrieval=dict(tags=memory.get('tags', []), limit_chars=memory.get('limit_chars', 4000)),
                budget=cfg['budget'], protected=[fingerprint(path.parent/p) for p in cfg.get('protected', [])]), root


def _task_copy(task, path, store, family):
    task = Path(task).resolve()
    cfg = read(task)
    cfg['plan'] = str((task.parent/cfg['plan']).resolve())
    cfg['protected'] = [str((task.parent/p).resolve()) for p in cfg.get('protected', [])]
    cfg['memory'] = dict(cfg.get('memory', {}), directory=str(store))
    cfg['memory'].pop('version', None)
    cfg['family'] = family
    if cfg['agent'].get('execution'):
        cfg['agent']['execution'] = str((task.parent/cfg['agent']['execution']).resolve())
    if path.exists() and read(path) != cfg:
        raise ValueError('benchmark task changed; use a new workspace')
    if not path.exists():
        write(path, cfg)
    return path


def run(suite, workspace, version_id, baseline_version, time_budget_seconds, construction_file=None):
    from .loop import _config, _positive, _version_key, run_task
    suite = Path(suite).resolve()
    spec = read(suite)
    if spec.get('schema_version') != 1 or not _positive(time_budget_seconds) or not spec.get('cases'):
        raise ValueError('benchmark requires cases and a positive time budget')
    store = (suite.parent/spec['store']).resolve()
    version = versions.load(store, version_id)
    if version['base_version'] != baseline_version:
        raise ValueError('benchmark baseline differs from candidate base')
    cases = spec['cases']
    if len({c['id'] for c in cases}) != len(cases) or any(not c.get('family') for c in cases):
        raise ValueError('benchmark requires unique IDs and task families')
    if set(version.get('evidence_families', [])) & {c['family'] for c in cases}:
        raise ValueError('benchmark overlaps improvement task families')
    reuse = spec.get('amortize_over_tasks', 1)
    if type(reuse) is not int or reuse <= 0:
        raise ValueError('positive amortize_over_tasks required')
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + time_budget_seconds
    with workspace_lock(workspace):
        state_path = workspace/'benchmark.json'
        binding = dict(suite=spec, version_id=version_id, baseline_version=baseline_version,
                       construction_file=str(construction_file) if construction_file else None)
        state = read(state_path) if state_path.exists() else dict(key=digest(binding), arms={}, conditions={})
        if state['key'] != digest(binding):
            raise ValueError('benchmark changed; use a new workspace')
        for index, case in enumerate(cases):
            tasks = {arm:(suite.parent/case[arm]).resolve() for arm in ('baseline','candidate') if arm in case}
            reused = read(suite.parent/case['baseline_receipt']) if case.get('baseline_receipt') else None
            if reused:
                tasks.pop('baseline', None)
            if str(index) not in state['conditions']:
                b, br = _signature(tasks['candidate'])
                if reused:
                    if (reused['version'] != baseline_version or reused['family'] != case['family'] or
                            reused['runtime'] != versions._runtime()):
                        raise ValueError('baseline receipt conditions are not comparable')
                    verify(reused['evidence'])
                    a, ar = reused['conditions'], Path(reused['root'])
                    state['arms'][str(index)+':baseline'] = reused
                else:
                    a, ar = _signature(tasks['baseline'])
                if ar == br or ar.is_relative_to(br) or br.is_relative_to(ar):
                    raise ValueError('benchmark arms need independent workload copies')
                if a != b:
                    raise ValueError('benchmark initial conditions differ')
                retrieval = b['retrieval']
                rejected = {arm:[] for arm in ('baseline','candidate')}
                selected = {arm:select(store,version=identity,diagnostics=rejected[arm],**retrieval)
                            for arm,identity in (('baseline',baseline_version),('candidate',version_id))}
                if selected['baseline']==selected['candidate']:
                    details = dict(case=case['id'],consumer=retrieval,rejected=rejected,
                                   selected={arm:[v['id'] for v in values] for arm,values in selected.items()})
                    write(workspace/'selection.json',details)
                    raise ValueError('candidate has no applicable runtime difference: ' + json.dumps(details))
                state['conditions'][str(index)] = a
                write(state_path,state)
            for arm, task in tasks.items():
                key = str(index)+':'+arm
                folder = workspace/str(index)/arm
                folder.mkdir(parents=True, exist_ok=True)
                configured = _task_copy(task, folder/'task.json', store, case['family'])
                if key in state['arms']:
                    verify(state['arms'][key]['evidence'])
                    recorded = read(folder/'session/loop.json')
                    if _version_key(_config(configured)[-1], recorded.get('system_version')) != recorded['key']:
                        raise ValueError('benchmark verification inputs changed')
                    continue
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise ValueError('benchmark time budget exhausted; resume with more budget')
                result = run_task(configured, folder/'session', time_budget_seconds=remaining,
                                  system_version=baseline_version if arm=='baseline' else version_id)
                evidence = collect([folder/'session'])['tasks'][0]
                state['arms'][key] = dict(correct=result['status']=='completed',
                    seconds=result['spent_seconds'], evidence=evidence,
                    usage=evidence['usage'], conditions=state['conditions'][str(index)],
                    family=case['family'], root=str(_config(task)[3]),
                    version=baseline_version if arm=='baseline' else version_id,
                    runtime=versions._runtime())
                write(folder/'result.json',state['arms'][key])
                write(state_path, state)
        values = {}
        for arm in ('baseline','candidate'):
            records = [state['arms'][str(i)+':'+arm] for i in range(len(cases))]
            values[arm] = dict(correct=all(r['correct'] for r in records),
                               seconds=sum(r['seconds'] for r in records),
                               usage=usage_totals(records))
        construction_path = construction_file or (suite.parent/spec['construction'] if spec.get('construction') else None)
        construction = read(construction_path) if construction_path else None
        if construction:
            if not _positive(construction.get('seconds')):
                raise ValueError('construction receipt needs measured positive seconds')
            values['candidate']['seconds'] += construction['seconds']/reuse
        a, b = values['baseline']['usage']['total_tokens'], values['candidate']['usage']['total_tokens']
        extra = (construction.get('usage') or {}).get('total_tokens') if construction else 0
        if all(type(v) is int for v in (a,b,extra)):
            values['baseline']['total_tokens'] = a
            values['candidate']['total_tokens'] = b + extra/reuse
        costs = {arm:price(values[arm]['usage'],spec.get('pricing')) for arm in values}
        extra_cost = construction.get('cost') if construction else 0
        if all(v is not None for v in costs.values()) and extra_cost is not None:
            values['baseline']['cost'] = costs['baseline']
            values['candidate']['cost'] = costs['candidate'] + extra_cost/reuse
        result = dict(status='completed',version_id=version_id, baseline_version=baseline_version, mode='closed',
            scope=[c['id'] for c in cases], families=sorted({c['family'] for c in cases}),
            baseline=values['baseline'], candidate=values['candidate'],
            construction=construction, amortize_over_tasks=reuse, evidence=str(state_path))
        write(workspace/'measurement.json', result)
        return result
