"""Shared literal-argv harness adapter; preserves raw output for usage accounting.

command(agent, prompt, run) writes prompt.txt and returns argv without a shell.
CLI: python harness.py AGENT.json PROMPT.txt RUN; execs the configured harness.
The CLI resolves the host-owned {run} placeholder in the prompt template.
"""
import json
import math
import os
from pathlib import Path
import shlex
import sys
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
            if not isinstance(event, dict) or event.get('type') != 'status' or event.get('phase') != 'step_end':
                continue
            values = event.get('usage', {})
            if not isinstance(values, dict):
                continue
            numbers = [values.get(k) for k in ('inputTokens', 'outputTokens', 'totalTokens')]
            if not all(type(v) is int and v >= 0 for v in numbers):
                continue
            for key, number in zip(('input_tokens', 'output_tokens', 'total_tokens'), numbers):
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


if __name__ == '__main__':
    agent = json.loads(Path(sys.argv[1]).read_text())
    prompt = Path(sys.argv[2]).read_text().replace('{run}', str(Path(sys.argv[3]).resolve()))
    (Path(sys.argv[3])/'artifacts').mkdir(exist_ok=True)
    argv = command(agent, prompt, sys.argv[3])
    os.execvpe(argv[0], argv, os.environ)
