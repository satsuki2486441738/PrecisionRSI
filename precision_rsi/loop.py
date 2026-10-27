"""Run a bounded model-driven task loop over existing native workflow tools.

CLI: python -m precision_rsi run|status TASK.json --workspace DIRECTORY
     python -m precision_rsi iterate SERIES.json --workspace DIRECTORY
     python -m precision_rsi evaluate ID --store DIRECTORY --plan PLAN --workspace DIRECTORY
Task JSON: schema_version=1, goal, plan, budget:{seconds,agent_calls},
agent:{argv:[...,"{prompt}" or "{prompt_file}"],timeout,execution?:profile},
protected:[verifier/input paths relative to the task JSON],
memory?:{directory,tags:[],limit_chars:4000}. Relative plan and memory paths are
relative to the task JSON. The native plan owns workload semantics and checks.
Run the loop where native commands and the selected agent harness can access them.
An existing harness (DeepSeek Harness, Codex, OpenCode, etc.) owns model tools,
provider configuration and model retries; this runner owns native verification,
time/call budgets, continuation and evidence reuse. Open operator search is reserved.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'skills/precision-compare/scripts'
sys.path.insert(0, str(TOOLS))

from record_run import record_run
from workflow import fingerprint, inspect_workflow, run_workflow, workspace_lock
from .memory import evaluate, propose, read, select, write


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _usage(path, format_name):
    """Normalize explicitly selected Harness step events; retain unknown cache usage."""
    if format_name is None:
        return None
    if format_name != 'dsh-jsonl':
        raise ValueError('unknown usage format')
    result = dict(input_tokens=0, output_tokens=0, total_tokens=0, steps=0,
                  cached_input_tokens=None)
    with Path(path).open(errors='replace') as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get('type') != 'status' or event.get('phase') != 'step_end':
                continue
            usage = event.get('usage', {})
            if not isinstance(usage, dict):
                continue
            values = [usage.get(k) for k in ('inputTokens', 'outputTokens', 'totalTokens')]
            if not all(type(v) is int and v >= 0 for v in values):
                continue
            for key, value in zip(('input_tokens', 'output_tokens', 'total_tokens'), values):
                result[key] += value
            result['steps'] += 1
    return result if result['steps'] else None


def _config(path):
    path = Path(path).resolve()
    cfg = read(path)
    if cfg.get('schema_version') != 1 or not isinstance(cfg.get('goal'), str) or not cfg['goal'].strip():
        raise ValueError('task schema_version=1 and goal required')
    if cfg.get('mode', 'closed') != 'closed':
        raise ValueError('open operator search awaits the operator-optimize implementation')
    budget, agent = cfg['budget'], cfg['agent']
    if agent.get('usage_format') not in (None, 'dsh-jsonl'):
        raise ValueError('unknown agent usage format')
    if not _positive(budget['seconds']) or type(budget['agent_calls']) is not int or budget['agent_calls'] < 0:
        raise ValueError('positive time budget and nonnegative agent call budget required')
    argv = agent['argv']
    if (not isinstance(argv, list) or not argv or not all(isinstance(a, str) and '\0' not in a for a in argv)
            or not any('{prompt}' in a or '{prompt_file}' in a for a in argv)
            or not _positive(agent['timeout'])):
        raise ValueError('agent argv needs a prompt placeholder and positive timeout')
    plan = (path.parent / cfg['plan']).resolve()
    spec = read(plan)
    root = (plan.parent / spec.get('root', '.')).resolve()
    protected = [plan] + [(path.parent / p).resolve() for p in cfg.get('protected', [])]
    identities = {str(p): fingerprint(p) for p in protected}
    semantic = {k: v for k, v in cfg.items() if k != 'budget'}
    key = hashlib.sha256(json.dumps(dict(task=semantic, protected=identities), sort_keys=True).encode()).hexdigest()
    return path, cfg, plan, root, identities, key


def _protected(identities):
    return all(fingerprint(p) == identity for p, identity in identities.items())


def _recover(workspace, state):
    for attempt in state['history']:
        if attempt['status'] != 'running':
            continue
        if attempt['kind'] == 'agent':
            for path in (workspace / 'agents').glob('compare-' + attempt['task'] + '-*/command.json'):
                command = read(path)
                pid = command.get('child_pid')
                if command['status'] == 'running' and pid:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise ValueError('previous agent may still be running: ' + str(path))
                attempt.update(run=str(path.parent), wall_seconds=command.get('wall_seconds') or attempt['wall_seconds'])
        attempt['status'] = 'interrupted'


def _prompt(cfg, plan, action, workspace, memories):
    skill = ROOT / 'skills' / action['skill'] / 'SKILL.md'
    action = {k:v for k,v in action.items() if k in
              ('stage','skill','reason','inputs','depends_on','outputs','run','execution')}
    if action.get('execution',{}).get('status') == 'completed':
        action.pop('execution')
    return ('Complete the indicated action in the existing workload. Load only the needed skill/interface. '
            'Preserve the task plan, protected inputs, evaluation criteria and workload semantics. '
            'Your delegated action ends when the implementation patch is saved. '
            'The host owns all workflow execution and original-workload checks. '
            'Reuse existing tools and evidence. Optional experience at the proposal path: '
            '{"kind":"context","content":"reusable rule","tags":[],"artifacts":[]}. '
            'Use kind=tool for reusable code, with absolute paths to its persistent files in artifacts. '
            'Use context for a reusable rule; no extra report is needed.\n'
            + json.dumps(dict(goal=cfg['goal'], plan=str(plan), skill=str(skill), action=action,
                              protected=cfg.get('protected', []), memory=memories,
                              proposal=str(workspace / 'proposal.json')), ensure_ascii=False))


def run_task(task, workspace, time_budget_seconds=None):
    path, cfg, plan, root, identities, key = _config(task)
    if time_budget_seconds is not None and not _positive(time_budget_seconds):
        raise ValueError('positive outer time budget required')
    deadline = None if time_budget_seconds is None else time.monotonic() + time_budget_seconds
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    state_path = workspace / 'loop.json'
    with workspace_lock(workspace):
        state = read(state_path) if state_path.exists() else dict(schema_version=1, task=str(path),
                key=key, history=[], created_at=datetime.now(timezone.utc).isoformat())
        if state.get('key') != key or state.get('task') != str(path):
            raise ValueError('task/verification changed; use a new workspace')
        _recover(workspace, state)
        memory = cfg.get('memory', {})
        store = (path.parent / memory['directory']).resolve() if memory.get('directory') else None
        report, reason, refresh, agent_failure = None, None, (), None
        while True:
            if not _protected(identities):
                reason = 'protected task or evaluator changed'
                break
            spent = sum(a['wall_seconds'] for a in state['history'])
            remaining = cfg['budget']['seconds'] - spent
            if deadline is not None:
                remaining = min(remaining, deadline - time.monotonic())
            if remaining <= 0:
                report = inspect_workflow(plan, workspace / 'native')
                reason = None if report['status'] == 'completed' else 'task time budget exhausted'
                break
            native = dict(kind='native', status='running', wall_seconds=remaining)
            state['history'].append(native)
            write(state_path, state)
            start = time.monotonic()
            try:
                report = run_workflow(plan, workspace / 'native', refresh, time_budget_seconds=remaining)
            finally:
                native.update(status='finished', wall_seconds=time.monotonic() - start)
                write(state_path, state)
            if not _protected(identities):
                reason = 'protected task or evaluator changed'
                break
            if report['status'] == 'completed':
                break
            if agent_failure:
                reason = agent_failure + '; native verification still incomplete'
                break
            remaining = cfg['budget']['seconds'] - sum(a['wall_seconds'] for a in state['history'])
            if deadline is not None:
                remaining = min(remaining, deadline - time.monotonic())
            calls = sum(a['kind'] == 'agent' for a in state['history'])
            if remaining <= 0 or calls >= cfg['budget']['agent_calls']:
                reason = 'task time budget exhausted' if remaining <= 0 else 'agent call budget exhausted'
                break
            if 'budget exhausted' in report['next_action']['reason']:
                reason = report['next_action']['reason']
                break
            native_state = read(workspace / 'native/state.json')
            matching = [a for a in native_state['history'] if a['stage'] == report['next_action']['stage']]
            action_key = dict(stage=report['next_action']['stage'], reason=report['next_action']['reason'],
                              identity=matching[-1]['key'] if matching else None)
            if state.get('last_action') == action_key:
                reason = 'agent made no verified progress'
                break
            state['last_action'] = action_key
            memories = select(store, memory.get('tags', []), memory.get('limit_chars', 4000)) if store else []
            prompt = _prompt(cfg, plan, report['next_action'], workspace, memories)
            timeout = min(cfg['agent']['timeout'], remaining)
            attempt = dict(kind='agent', task='agent-call' + str(calls + 1), status='running',
                           wall_seconds=timeout, memory_ids=[m['id'] for m in memories])
            state['history'].append(attempt)
            write(state_path, state)
            def argv(run):
                prompt_file = run / 'prompt.txt'
                prompt_file.write_text(prompt)
                return [a.replace('{prompt}', prompt).replace('{prompt_file}', str(prompt_file))
                        for a in cfg['agent']['argv']]
            execution = cfg['agent'].get('execution')
            folder, command = record_run(workspace / 'agents', 'compare', attempt['task'], timeout,
                    argv, cwd=root, execution=(path.parent / execution).resolve() if execution else None)
            attempt.update(status=command['status'], wall_seconds=command['wall_seconds'], run=str(folder),
                           usage=_usage(folder/'stdout.bin',cfg['agent'].get('usage_format')),
                           raw_output=str(folder / 'stdout.bin'))
            write(state_path, state)
            if command['status'] != 'completed':
                agent_failure = 'agent ' + command['status']
            refresh = (report['next_action']['stage'],)
        completed = reason is None and report is not None and report['status'] == 'completed'
        result = dict(status='completed' if completed else 'needs_action', reason=reason,
                      workflow=report, agent_calls=sum(a['kind'] == 'agent' for a in state['history']),
                      spent_seconds=sum(a['wall_seconds'] for a in state['history']),
                      state=str(state_path), candidate=None)
        proposal = workspace / 'proposal.json'
        if completed and store and proposal.is_file():
            # Final independent native checks, rather than the agent's completion claim,
            # establish the source task. Admission needs a separate evaluation.
            try:
                result['candidate'] = propose(store, read(proposal), dict(task=str(path),
                        key=key, workflow=str(workspace / 'native/state.json')))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result['candidate_error'] = str(exc)
        state['result'] = result
        write(state_path, state)
        return result


def _render(value, replacements):
    if isinstance(value, str):
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        return value
    if isinstance(value, list):
        return [_render(x, replacements) for x in value]
    if isinstance(value, dict):
        return {k:_render(v, replacements) for k,v in value.items()}
    return value


def iterate(series, workspace):
    """Run a finite task stream and evaluate each new candidate before later tasks.

    Series JSON: schema_version=1, tasks:[task_JSON_paths], budget_seconds,
    evaluation_plan?:workflow_template. The evaluator template uses literal
    {candidate_id}/{candidate_file} substitutions; root stays relative to its
    original location. Every source task needs a memory directory to propose.
    Task and evaluator receipts support replay without rerunning unchanged work.
    """
    path = Path(series).resolve()
    spec = read(path)
    if (spec.get('schema_version') != 1 or not _positive(spec.get('budget_seconds'))
            or not isinstance(spec.get('tasks'), list) or not spec['tasks']
            or not all(isinstance(t, str) and t for t in spec['tasks'])):
        raise ValueError('series needs tasks and a positive time budget')
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    with workspace_lock(workspace):
        state_path = workspace / 'series.json'
        key = hashlib.sha256(json.dumps({k:v for k,v in spec.items() if k!='budget_seconds'},
                                        sort_keys=True).encode()).hexdigest()
        state = read(state_path) if state_path.exists() else dict(key=key, history=[])
        if state['key'] != key:
            raise ValueError('series changed; use a new workspace')
        # Reserve the remaining budget before work, preserving a bound after interruption.
        remaining = spec['budget_seconds'] - sum(a['seconds'] for a in state['history'])
        if remaining <= 0:
            return dict(status='needs_action', reason='series time budget exhausted',
                        tasks=[], spent_seconds=spec['budget_seconds']-remaining)
        attempt = dict(seconds=remaining, status='running')
        state['history'].append(attempt)
        write(state_path,state)
        start = time.monotonic()
        results = []
        deadline = start + remaining
        try:
            for index, name in enumerate(spec['tasks']):
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    break
                task = (path.parent/name).resolve()
                task_workspace = workspace/'tasks'/str(index)
                result = run_task(task, task_workspace, time_budget_seconds=remaining)
                entry = dict(task=str(task), result=result, evaluation=None)
                results.append(entry)
                candidate = result.get('candidate')
                remaining = deadline-time.monotonic()
                if candidate and spec.get('evaluation_plan') and remaining > 0:
                    cfg = read(task)
                    store = (task.parent/cfg['memory']['directory']).resolve()
                    template = (path.parent/spec['evaluation_plan']).resolve()
                    evaluation = _render(read(template), {'{candidate_id}':candidate,
                        '{candidate_file}':str(store/(candidate+'.json'))})
                    evaluation['root'] = str((template.parent/evaluation.get('root','.')).resolve())
                    evaluation_path = task_workspace/'evaluation-plan.json'
                    write(evaluation_path,evaluation)
                    entry['evaluation'] = evaluate(store,candidate,evaluation_path,
                            task_workspace/'evaluation',time_budget_seconds=remaining)
            result = dict(status='completed' if len(results)==len(spec['tasks']) and
                          all(x['result']['status']=='completed' for x in results) else 'needs_action',
                          tasks=results, state=str(state_path))
        finally:
            attempt.update(seconds=time.monotonic()-start,status='finished')
            write(state_path,state)
        result['spent_seconds'] = sum(a['seconds'] for a in state['history'])
        state['result'] = result
        write(state_path,state)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'status', 'iterate'):
        sub = commands.add_parser(name)
        sub.add_argument('task')
        sub.add_argument('--workspace', required=True)
    sub = commands.add_parser('evaluate')
    sub.add_argument('candidate_id')
    sub.add_argument('--store', required=True)
    sub.add_argument('--plan', required=True)
    sub.add_argument('--workspace', required=True)
    sub.add_argument('--time-budget-seconds', type=float, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            result = run_task(args.task, args.workspace)
        elif args.command == 'iterate':
            result = iterate(args.task, args.workspace)
        elif args.command == 'status':
            _, _, plan, _, _, key = _config(args.task)
            state_path = Path(args.workspace) / 'loop.json'
            state = read(state_path) if state_path.exists() else None
            result = dict(status='new' if state is None else 'recorded',
                          identity_matches=state is None or key == state.get('key'),
                          previous=None if state is None else state.get('result'),
                          workflow=inspect_workflow(plan, Path(args.workspace) / 'native'))
        else:
            result = evaluate(args.store, args.candidate_id, args.plan, args.workspace, args.time_budget_seconds)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps(dict(status='error', reason=str(exc))))
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0 if result['status'] in ('completed', 'active', 'recorded', 'new') else 1
