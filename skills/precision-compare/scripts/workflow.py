#!/usr/bin/env python3
"""Run and resume dependency-linked adaptation/comparison/repair commands (POSIX, stdlib).
Call: run_workflow(plan_path, workspace, refresh=()); inspect_workflow(plan_path, workspace)
Input: Dependency plan with source/runtime inputs, outputs, checks and cumulative budget.
Output: Verified reuse/execution receipts, artifact paths and next_action.
Requires: Standard library/POSIX; status launches nothing; unchanged failures are retained; --refresh reruns selected stages and descendants.

Details:
API: run_workflow(plan_path, workspace, refresh=()) or inspect_workflow(plan_path,
workspace) -> compact continuation report. CLI: workflow.py run|status --plan P
--workspace W [--refresh STAGE ...]. Exit 0=all declared stages completed,
1=next action/budget required, 2=invalid plan or concurrent runner.

Plan JSON: schema_version=1, id (alphanumeric/hyphens), root (relative to plan,
optional '.'), workload:optional object (model/task/target setups), budget_seconds (cumulative child wall time, including retries),
stages=[{id, skill:framework-adapt|precision-compare|precision-repair,
argv:[literal,...], inputs:[file_or_directory,...], outputs:{name:relative_path},
depends_on:[stage_id,...], timeout:seconds, side:optional 'compare',
cache:optional true, env_keys:optional [inherited_env_name,...],
checks:optional [{output:name,pointer:'/JSON/pointer',equals:JSON_value}],
on_check_failure:optional skill}]. Inputs cover source, model/data and runtime
identity manifests relevant to reuse; remote adapters supply local identity and
result receipts. Hashing follows declared dependencies, not undeclared changes
inside remote hosts or installed packages. Select only relevant files/directories.

argv substitutes {run} (fresh stage directory), {root}, and
{artifact:STAGE:NAME} (a declared dependency output), without shell evaluation.
Outputs must be files/directories inside that run. Missing outputs, failed JSON
checks, changing inputs and unsuccessful commands are never successful cache hits.
No precision tolerance is supplied: checks express task-owned validation or
correctness evidence; comparator exit 0 establishes metrics, not user acceptance.

Workspace state.json contains attempts, content identities and artifact paths.
Successful unchanged stages are reused after checking output hashes. A matching
failure yields its continuation without rerunning; change its inputs or request
--refresh. Refresh reruns the stage and descendants. cache:false stages execute
on each run (e.g. a live environment probe); unchanged receipts permit downstream
reuse. Failures route to the stage skill; a failed task-owned check may route to
on_check_failure. The agent performs the indicated adaptation/repair, then calls
run again to verify the affected dependency chain. No model service is invoked.

Commands use record_run for literal argv, logs, timeout and process-group cleanup.
Existing run bytes are preserved. A process lock prevents concurrent execution.
After a hard-killed runner, a still-live recorded child blocks duplicate launch.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

from conditions import canonical, load_json
from record_run import record_run

SKILLS = {'framework-adapt', 'precision-compare', 'precision-repair'}
ID = re.compile(r'[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\Z')
TOKEN = re.compile(r'\{artifact:([^:{}]+):([^:{}]+)\}')
ENGINE_FILES = tuple(Path(__file__).with_name(name) for name in ('workflow.py', 'record_run.py', 'conditions.py'))
ENV_KEYS = ('PATH', 'PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH',
            'CUDA_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES',
            'ASCEND_HOME_PATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def fingerprint(path, symlinks=True, ancestors=frozenset()):
    """Content identity for selected local files/directories, detecting changing files."""
    path = Path(path)
    _require(symlinks or not path.is_symlink(), 'output identities require literal files')
    _require(path.exists(), 'missing input/output: ' + str(path))
    if path.is_dir():
        real = path.resolve()
        _require(real not in ancestors, 'directory identity contains a symbolic-link cycle')
        entries = [(p.name, fingerprint(p, symlinks, ancestors | {real}))
                   for p in sorted(path.iterdir())]
        return dict(kind='directory', sha256=_digest(entries),
                    files=sum(v.get('files', 1) for _, v in entries),
                    bytes=sum(v['bytes'] for _, v in entries))
    _require(path.is_file(), 'expected regular file: ' + str(path))
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    after = path.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    _require(stamp(before) == stamp(after), 'file changed while hashing: ' + str(path))
    return dict(kind='file', sha256=h.hexdigest(), bytes=after.st_size)


def _plan(path):
    path = Path(path).resolve()
    spec = load_json(path)
    _require(isinstance(spec, dict) and spec.get('schema_version') == 1, 'plan schema_version must be 1')
    _require(isinstance(spec.get('id'), str) and ID.fullmatch(spec['id']), 'invalid workflow id')
    _require(_positive(spec.get('budget_seconds')), 'positive budget_seconds required')
    _require(isinstance(spec.get('workload', {}), dict), 'workload must be an object')
    root = (path.parent / spec.get('root', '.')).resolve()
    _require(root.is_dir(), 'workflow root must exist')
    stages = spec.get('stages')
    _require(isinstance(stages, list) and stages, 'nonempty stages required')
    by_id = {}
    for raw in stages:
        _require(isinstance(raw, dict), 'stage must be an object')
        stage = dict(raw)
        name = stage.get('id')
        _require(isinstance(name, str) and ID.fullmatch(name) and name not in by_id, 'unique stage IDs required')
        _require(stage.get('skill') in SKILLS, 'unknown stage skill')
        _require(stage.get('on_check_failure', stage['skill']) in SKILLS, 'unknown check-failure skill')
        argv = stage.get('argv')
        _require(isinstance(argv, list) and argv and all(isinstance(x, str) and '\0' not in x for x in argv) and argv[0], 'literal argv required')
        _require(_positive(stage.get('timeout')), 'positive stage timeout required')
        for field in ('inputs', 'depends_on', 'env_keys'):
            stage.setdefault(field, [])
            _require(isinstance(stage[field], list) and all(isinstance(x, str) and x for x in stage[field])
                     and len(set(stage[field])) == len(stage[field]), field + ' must contain unique strings')
        stage.setdefault('side', 'compare')
        _require(isinstance(stage['side'], str) and ID.fullmatch(stage['side']), 'invalid stage side')
        stage.setdefault('cache', True)
        _require(type(stage['cache']) is bool, 'cache must be boolean')
        outputs = stage.get('outputs')
        _require(isinstance(outputs, dict) and outputs, 'named outputs required')
        for key, value in outputs.items():
            _require(isinstance(key, str) and ID.fullmatch(key), 'invalid output name')
            _require(isinstance(value, str) and value and not Path(value).is_absolute()
                     and '..' not in Path(value).parts and value != '.', 'output must be a relative run path')
        _require(len(set(outputs.values())) == len(outputs), 'output paths must be distinct')
        stage.setdefault('checks', [])
        _require(isinstance(stage['checks'], list), 'checks must be a list')
        for check in stage['checks']:
            _require(isinstance(check, dict) and set(check) == {'output', 'pointer', 'equals'}
                     and check['output'] in outputs and isinstance(check['pointer'], str)
                     and (not check['pointer'] or check['pointer'].startswith('/')), 'invalid JSON output check')
        by_id[name] = stage
    for name, stage in by_id.items():
        _require(all(dep in by_id and dep != name for dep in stage['depends_on']), 'unknown/self dependency')
        for arg in stage['argv']:
            for dep, output in TOKEN.findall(arg):
                _require(dep in stage['depends_on'] and output in by_id[dep]['outputs'], 'artifact must name a direct dependency output')
    ordered, visiting, visited = [], set(), set()
    def visit(name):
        _require(name not in visiting, 'cyclic stage dependency')
        if name in visited:
            return
        visiting.add(name)
        for dep in by_id[name]['depends_on']:
            visit(dep)
        visiting.remove(name)
        visited.add(name)
        ordered.append(by_id[name])
    for name in by_id:
        visit(name)
    return spec, root, ordered


def _state(workspace, spec, root):
    path = workspace / 'state.json'
    if not path.exists():
        return dict(schema_version=1, id=spec['id'], root=str(root), history=[])
    state = load_json(path)
    _require(state.get('schema_version') == 1 and state.get('id') == spec['id']
             and state.get('root') == str(root) and isinstance(state.get('history'), list),
             'workspace belongs to a different workflow/root')
    return state


def _save(workspace, state):
    temporary = workspace / 'state.json.tmp'
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    temporary.replace(workspace / 'state.json')


@contextmanager
def workspace_lock(workspace):
    """Exclusively lock an existing POSIX workspace without waiting or unlink races."""
    _require(os.name == 'posix', 'workflow command runner requires POSIX process locking')
    import fcntl
    with (workspace / 'runner.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('workflow already has an active runner') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _signature(stage, root, completed, workload):
    inputs = {name: fingerprint(root / name) for name in stage['inputs']}
    deps = {}
    for name in stage['depends_on']:
        outputs = {}
        for key, value in completed[name]['outputs'].items():
            actual = fingerprint(value['path'], symlinks=False)
            _require(actual == value['identity'], 'dependency output changed: ' + name + ':' + key)
            outputs[key] = actual
        deps[name] = dict(key=completed[name]['key'], outputs=outputs)
    candidate = root / stage['argv'][0]
    executable = (str(candidate.resolve()) if candidate.is_file() else None) if '/' in stage['argv'][0] else shutil.which(stage['argv'][0])
    keys = sorted(set(ENV_KEYS) | set(stage['env_keys']))
    environment = _digest({k: os.environ.get(k) for k in keys})
    return dict(stage=stage, root=str(root), workload=workload, inputs=inputs, dependencies=deps,
                execution_tools={p.name: fingerprint(p) for p in ENGINE_FILES},
                executable=fingerprint(executable) if executable else None, environment=environment)


def _valid(receipt):
    try:
        return receipt['status'] == 'completed' and all(
            not Path(v['path']).is_symlink()
            and Path(v['path']).resolve().is_relative_to(Path(receipt['run']).resolve())
            and fingerprint(v['path'], symlinks=False) == v['identity'] for v in receipt['outputs'].values())
    except (KeyError, OSError, ValueError):
        return False


def _latest(state, name):
    return next((r for r in reversed(state['history']) if r['stage'] == name), None)


def _alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _recover(workspace, state):
    # Running receipts were persisted before launch. A hard-killed parent can leave
    # a native child; never start the same work while that child still exists.
    for receipt in state['history']:
        if receipt['status'] != 'running':
            continue
        folders = list((workspace / 'runs').glob(receipt['run_prefix'] + '*'))
        for folder in folders:
            command = folder / 'command.json'
            if command.exists():
                metadata = load_json(command)
                _require(metadata['status'] != 'running' or not _alive(metadata.get('child_pid')),
                         'previous native child may still be running: ' + str(command))
                receipt.update(run=str(folder), wall_seconds=metadata.get('wall_seconds') or receipt['timeout'])
        receipt.update(status='interrupted', reason='previous runner ended before recording verified outputs')


def _argv(stage, root, run, completed):
    def replace(match):
        return completed[match[1]]['outputs'][match[2]]['path']
    return [TOKEN.sub(replace, arg).replace('{root}', str(root)).replace('{run}', str(run))
            for arg in stage['argv']]


def _pointer(document, pointer):
    for part in pointer.split('/')[1:] if pointer else []:
        _require(not re.search(r'~(?![01])', part), 'invalid JSON pointer escape')
        part = part.replace('~1', '/').replace('~0', '~')
        if isinstance(document, list):
            _require(bool(re.fullmatch(r'0|[1-9][0-9]*', part)), 'invalid JSON pointer array index')
            document = document[int(part)]
        else:
            document = document[part]
    return document


def _outputs(stage, run):
    result = {}
    for name, relative in stage['outputs'].items():
        path = run / relative
        _require(not path.is_symlink() and path.resolve().is_relative_to(run.resolve()), 'output escaped stage run')
        result[name] = dict(path=str(path.resolve()), identity=fingerprint(path, symlinks=False))
    return result


class CheckFailed(ValueError):
    """A well-formed task-owned predicate disagrees with a captured value."""


def _checks(stage, outputs):
    for check in stage['checks']:
        actual = _pointer(load_json(outputs[check['output']]['path']), check['pointer'])
        if canonical(actual) != canonical(check['equals']):
            raise CheckFailed('output check failed: ' + check['output'] + check['pointer'])


def _report(spec, state, entries, next_action=None):
    return dict(id=spec['id'], workload=spec.get('workload', {}), status='needs_action' if next_action else 'completed',
                spent_seconds=sum(x.get('wall_seconds', 0) for x in state['history']),
                budget_seconds=spec['budget_seconds'], attempts=len(state['history']),
                stages=entries, next_action=next_action)


def _action(stage, reason, receipt=None):
    result = dict(stage=stage['id'], skill=(receipt or {}).get('next_skill', stage['skill']),
                  reason=reason, inputs=stage['inputs'], depends_on=stage['depends_on'])
    if receipt and receipt.get('run'):
        result.update(run=receipt['run'], stdout=str(Path(receipt['run']) / 'stdout.bin'),
                      stderr=str(Path(receipt['run']) / 'stderr.bin'), outputs=receipt.get('outputs', {}))
    return result


def _walk(spec, root, stages, workspace, state, execute, refresh):
    completed, entries = {}, []
    forced = set(refresh)
    _require(forced <= {s['id'] for s in stages}, 'unknown refresh stage')
    for stage in stages:
        name = stage['id']
        if any(dep in forced for dep in stage['depends_on']):
            forced.add(name)
        try:
            identity = _signature(stage, root, completed, spec.get('workload', {}))
            key = _digest(identity)
        except (OSError, ValueError) as exc:
            return _report(spec, state, entries, _action(stage, str(exc)))
        latest = _latest(state, name)
        same = latest and latest['key'] == key
        if stage['cache'] and name not in forced and same and _valid(latest):
            completed[name] = latest
            entries.append(dict(id=name, skill=stage['skill'], reused=True, run=latest['run'], outputs=latest['outputs']))
            continue
        if not execute:
            return _report(spec, state, entries, _action(stage, 'stage needs execution' if not same else latest.get('reason', 'cached outputs unavailable'), latest))
        if same and name not in forced and latest['status'] != 'completed':
            return _report(spec, state, entries, _action(stage, latest.get('reason', latest['status']), latest))
        spent = sum(x.get('wall_seconds', 0) for x in state['history'])
        remaining = spec['budget_seconds'] - spent
        if remaining <= 0:
            return _report(spec, state, entries, _action(stage, 'cumulative command budget exhausted', latest))
        timeout = min(stage['timeout'], remaining)
        task = f"{spec['id']}-{name}-attempt{len(state['history']) + 1}"
        receipt = dict(stage=name, skill=stage['skill'], key=key, identity=identity, status='running', outputs={},
                       run_prefix=f"{stage['side']}-{task}-", timeout=timeout, wall_seconds=0)
        state['history'].append(receipt)
        _save(workspace, state)
        folder, command = record_run(workspace / 'runs', stage['side'], task, timeout,
                                     lambda run: _argv(stage, root, run, completed), root)
        receipt.update(run=str(folder), status=command['status'], wall_seconds=command['wall_seconds'],
                       returncode=command['returncode'])
        if command['status'] == 'completed':
            try:
                receipt['outputs'] = _outputs(stage, folder)
                _require(_digest(_signature(stage, root, completed, spec.get('workload', {}))) == key, 'declared inputs changed during execution')
                try:
                    _checks(stage, receipt['outputs'])
                except CheckFailed as exc:
                    receipt['next_skill'] = stage.get('on_check_failure', stage['skill'])
                    raise ValueError(str(exc)) from exc
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                receipt.update(status='invalid_outputs', reason=str(exc))
        else:
            receipt['reason'] = 'native command ' + command['status']
        _save(workspace, state)
        if receipt['status'] != 'completed':
            return _report(spec, state, entries, _action(stage, receipt.get('reason', receipt['status']), receipt))
        completed[name] = receipt
        entries.append(dict(id=name, skill=stage['skill'], reused=False, run=str(folder), outputs=receipt['outputs']))
    return _report(spec, state, entries)


def run_workflow(plan_path, workspace, refresh=()):
    """Execute missing stages and return their reusable outputs or the next skill action."""
    spec, root, stages = _plan(plan_path)
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    with workspace_lock(workspace):
        state = _state(workspace, spec, root)
        _recover(workspace, state)
        result = _walk(spec, root, stages, workspace, state, True, refresh)
        _save(workspace, state)
        return result


def inspect_workflow(plan_path, workspace):
    """Inspect reuse and the next required action without executing or creating files."""
    spec, root, stages = _plan(plan_path)
    workspace = Path(workspace).resolve()
    state = _state(workspace, spec, root)
    return _walk(spec, root, stages, workspace, state, False, ())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.partition('\n\nDetails:\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('run', 'status'))
    parser.add_argument('--plan', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--refresh', action='append', default=[])
    args = parser.parse_args(argv)
    try:
        _require(args.command == 'run' or not args.refresh, '--refresh applies to run')
        report = (run_workflow(args.plan, args.workspace, args.refresh) if args.command == 'run'
                  else inspect_workflow(args.plan, args.workspace))
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        print(json.dumps(dict(status='error', reason=str(exc))))
        return 2
    print(json.dumps(report, allow_nan=False))
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
