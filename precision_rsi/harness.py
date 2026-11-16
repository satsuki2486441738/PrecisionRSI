"""Shared literal-argv harness adapter; preserves raw output for usage accounting.

command(agent, prompt, run) writes prompt.txt and returns argv without a shell.
CLI: python harness.py AGENT.json PROMPT.txt RUN [--cwd-run]; execs the configured harness.
The CLI resolves the host-owned {run} placeholder in the prompt template.
--cwd-run places relative agent outputs in RUN; task workloads retain their cwd.
"""
import json
import math
import os
from pathlib import Path
import shlex
import sys
import subprocess
import shutil
from collections import Counter


def command(agent, prompt, run):
    argv = agent['argv']
    if (not isinstance(argv, list) or not argv or
            not all(isinstance(a, str) and '\0' not in a for a in argv) or
            not any('{prompt}' in a or '{prompt_file}' in a for a in argv)):
        raise ValueError('agent argv needs a prompt placeholder')
    prompt_file = Path(run)/'prompt.txt'
    prompt_file.write_text(prompt)
    return [a.replace('{prompt}', prompt).replace('{prompt_file}', str(prompt_file)) for a in argv]


def _step_usage(event):
    if not isinstance(event, dict) or event.get('type') != 'status' or event.get('phase') != 'step_end':
        return None
    values = event.get('usage', {})
    if not isinstance(values, dict):
        return None
    numbers = [values.get(k) for k in ('inputTokens', 'outputTokens', 'totalTokens')]
    if not all(type(v) is int and v >= 0 for v in numbers):
        return None
    return dict(zip(('input_tokens', 'output_tokens', 'total_tokens'), numbers))


def usage(path, format_name):
    """Count completed DSH step events; unknown cache usage stays null."""
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
            measured = _step_usage(event)
            if measured is None:
                continue
            for key, number in measured.items():
                result[key] += number
            result['steps'] += 1
    return result if result['steps'] else None


def activity(path, format_name):
    """Bounded DSH tool-call counts; filenames/mentions, never command or file contents.

    Python mentions describe shell tokens, not proven executions. Repeated
    commands count identical bash inputs. Duplicate call IDs count once.
    """
    if format_name is None:
        return None
    if format_name != 'dsh-jsonl':
        raise ValueError('unknown activity format')
    tools, reads, mentions, commands = (Counter() for _ in range(4))
    seen = set()
    with Path(path).open(errors='replace') as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get('type') != 'tool_call':
                continue
            tool, args, identity = event.get('tool'), event.get('input'), event.get('callId')
            if not isinstance(tool, str) or not isinstance(args, dict):
                continue
            if isinstance(identity, str):
                if identity in seen:
                    continue
                seen.add(identity)
            tools[tool[:80]] += 1
            if tool == 'read' and isinstance(args.get('file_path'), str):
                reads[Path(args['file_path']).name[:120]] += 1
            if tool == 'bash' and isinstance(args.get('command'), str):
                command = args['command']
                commands[command] += 1
                try:
                    tokens = shlex.split(command)
                except ValueError:
                    tokens = []
                for token in tokens:
                    token = token.rstrip(';|&')
                    if token.endswith('.py') and not any(c in token for c in '\n*?['):
                        mentions[Path(token).name[:120]] += 1
    if not tools:
        return None
    return dict(tool_calls=sum(tools.values()), tools=dict(tools.most_common(16)),
                read_files=dict(reads.most_common(12)), python_mentions=dict(mentions.most_common(12)),
                repeated_commands=sum(n-1 for n in commands.values()))


def price(usage, rates):
    """Price supplied per-million rates; missing usage/cache attribution stays unknown."""
    if rates is None or usage is None:
        return None
    incoming, outgoing = rates['input_per_million'], rates['output_per_million']
    cached = rates.get('cached_input_per_million', incoming)
    if not all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in (incoming,outgoing,cached)):
        raise ValueError('pricing rates must be finite and nonnegative')
    inputs, outputs, hits = (usage.get(k) for k in ('input_tokens','output_tokens','cached_input_tokens'))
    if not all(type(v) is int and v >= 0 for v in (inputs,outputs)):
        return None
    if hits is None:
        if cached != incoming:
            return None
        hits = 0
    if type(hits) is not int or not 0 <= hits <= inputs:
        raise ValueError('cached input count must be within total input count')
    return ((inputs-hits)*incoming + hits*cached + outputs*outgoing)/1_000_000


def persistent(agent, prompt, run):
    """Resume one native DSH session; receipts prevent duplicate turn execution.

    agent.research: {root,binding,turn}; agent.persistence: {kind:'dsh'}.
    argv must forward DSH flags and have a final, standalone prompt placeholder.
    Native cwd is stable; candidate output paths in prompt remain absolute.
    An interrupted turn blocks until an operator explicitly reconciles it.
    """
    # Also works when invoked as a script by workflow.py.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from precision_rsi.research import Research
    from precision_rsi.memory import fingerprint, read, workspace_lock, write
    info = agent['research']
    research = Research(info['root'], info['binding'])
    turn = info['turn']
    policy = agent.get('persistence')
    if policy is not None and policy != {'kind': 'dsh'}:
        raise ValueError('supported persistence is {kind:dsh}')
    if policy and (agent.get('usage_format') != 'dsh-jsonl' or '--json' not in agent['argv'] or
                   agent['argv'][-1] not in ('{prompt}', '{prompt_file}')):
        raise ValueError('DSH persistence requires dsh-jsonl and a final standalone prompt')
    with workspace_lock(research.private):
        binding = dict(agent=agent, prompt=prompt, run=str(run))
        cached = research.begin(turn, binding)
        if cached:
            if cached.get('status') == 'interrupted':
                raise ValueError('turn explicitly interrupted; use a new turn')
            if fingerprint(cached['stdout']) != cached['stdout_identity']:
                raise ValueError('researcher receipt output changed')
            with Path(cached['stdout']).open('rb') as stream:
                shutil.copyfileobj(stream, sys.stdout.buffer)
            return cached['returncode']
        research.checkpoint(turn, run)
        argv = command(agent, prompt, run)
        cwd = Path(run)
        cursor_path = research.private/'cursor.json'
        cursor = read(cursor_path) if cursor_path.exists() else None
        if policy:
            cwd = research.view/'work'
            cwd.mkdir(exist_ok=True)
            patch = research.private/'persistence.patch.json'
            write(patch, [{'id':'session-persistence-jsonl', 'config':{
                'root':str(research.private/'native'), 'compression':'none'}}])
            if cursor is None and any((research.private/'native').rglob('*.jsonl')):
                raise ValueError('native session cursor missing; explicit recovery required')
            if cursor:
                for name, identity in cursor['logs'].items():
                    path = research.private/name
                    if not path.is_file() or fingerprint(path) != identity:
                        raise ValueError('native session missing or changed; explicit recovery required')
                if not cursor['logs']:
                    raise ValueError('native session cursor has no logs')
                argv[-1:-1] = ['--session-id', cursor['session_id']]
            # --json starts profile passthrough: global overlays must precede it.
            # Place ours after caller overlays so a model patch cannot reset root.
            position = argv.index('--json')
            argv[position:position] = ['--patch', str(patch)]
        env = dict(os.environ)
        root = str(Path(__file__).resolve().parents[1])
        env['PYTHONPATH'] = root + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        spool = research.private/('turn-'+turn+'.stdout.bin')
        session = None
        observed = None
        with spool.open('wb') as output:
            with subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE) as process:
                for line in process.stdout:
                    output.write(line)
                    output.flush()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        event = None
                    if policy and isinstance(event, dict) and event.get('type') == 'session':
                        session = event.get('sessionId') or event.get('id')
                    measured = _step_usage(event) if agent.get('usage_format') == 'dsh-jsonl' else None
                    if measured is not None:
                        if observed is None:
                            observed = dict(input_tokens=0, output_tokens=0, total_tokens=0,
                                            steps=0, cached_input_tokens=None)
                        for key, value in measured.items():
                            observed[key] += value
                        observed['steps'] += 1
                        research.checkpoint(turn, run, usage=observed)
                    # Checkpoint first: a parent that observes step_end can lose
                    # the entire process group without losing the completed step.
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                code = process.wait()
        if policy:
            logs = {p.relative_to(research.private).as_posix():fingerprint(p)
                    for p in (research.private/'native').rglob('*.jsonl')}
            if not isinstance(session, str) or not session or not logs or (
                    cursor and session != cursor['session_id']):
                raise ValueError('DSH did not retain the expected native session; reconcile turn')
            write(cursor_path, dict(session_id=session, logs=logs))
        research.complete(turn, status='completed' if code == 0 else 'failed', returncode=code,
            stdout=str(spool), stdout_identity=fingerprint(spool),
            usage=usage(spool, agent.get('usage_format')))
        research.checkpoint(turn, run, status='completed' if code == 0 else 'failed', usage=observed)
        return code


if __name__ == '__main__':
    agent = json.loads(Path(sys.argv[1]).read_text())
    run = Path(sys.argv[3]).resolve()
    prompt = Path(sys.argv[2]).read_text().replace('{run}', str(run))
    (run/'artifacts').mkdir(exist_ok=True)
    if agent.get('research'):
        sys.exit(persistent(agent, prompt, run))
    argv = command(agent, prompt, run)
    if sys.argv[4:]:
        if sys.argv[4:] != ['--cwd-run']:
            raise ValueError('expected --cwd-run')
        os.chdir(run)
    os.execvpe(argv[0], argv, os.environ)
