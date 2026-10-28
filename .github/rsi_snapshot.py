"""Export a public RSI dashboard snapshot from private run evidence.

python .github/rsi_snapshot.py --monitor projects/rsi-system/monitor.json \
    --output site/data/rsi.json [--refresh]

--refresh reads the current remote run over the monitor's SSH connection.
Only allowlisted summaries are returned; traces, paths and credentials stay private.
"""
import argparse
import collections
import datetime
import inspect
import json
import math
from pathlib import Path
import re
import subprocess


def summarize(directory):
    """Read native receipts; return presentation data, never full traces."""
    root = Path(directory)
    def read(name):
        for base in (root, root / 'evidence'):
            path = base / name
            if path.is_file():
                return json.loads(path.read_text())
        return {}
    def number(value):
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None
    def clean(value):
        text = str(value or '')
        text = re.sub(r'(?:/[\w.()~+-]+){2,}', '[private path]', text)
        text = re.sub(r'(?i)(?:bearer\s+|(?:api[_-]?key|token|password)\s*[:=]\s*)\S+', '[redacted]', text)
        return text[:400]
    def usage(value):
        value = value if isinstance(value, dict) else {}
        return {key: number(value.get(key)) for key in ('input_tokens', 'output_tokens', 'total_tokens', 'cached_input_tokens')}
    def phase(value):
        value = value if isinstance(value, dict) else {}
        return dict(seconds=number(value.get('seconds')), usage=usage(value.get('usage')))
    def activity(paths):
        tools = collections.Counter()
        for path in paths:
            with path.open(errors='replace') as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get('type') == 'tool_call':
                        tool = event.get('tool')
                        name = event.get('name') or (tool.get('name') if isinstance(tool, dict) else tool)
                        if isinstance(name, str):
                            tools[clean(name)] += 1
        return dict(tool_calls=sum(tools.values()), tools=dict(tools)) if paths else None
    status, evolution, experiment = read('status.json'), read('evolution/evolution.json'), read('experiment.json')
    audit = read('audit.json')
    if not status:
        return None
    result = evolution.get('result') or status.get('evolution') or {}
    raw = read('raw-usage.json')
    baseline = status.get('baseline') or {}
    phases = {}
    for name in ('baseline', 'construction', 'evaluation'):
        if name == 'baseline':
            phases[name] = phase(baseline)
            if baseline.get('reused'):
                phases[name] = dict(seconds=0, usage=dict(input_tokens=0, output_tokens=0, total_tokens=0, cached_input_tokens=None))
        else:
            recorded = (audit.get('phases') or {}).get(name) or {}
            phases[name] = phase(recorded)
            phases[name]['usage'] = usage((status.get('raw_usage') or {}).get(name) or (raw.get(name) or {}).get('usage') or recorded.get('usage') or result.get(name + '_usage'))
    rounds = []
    for i, item in enumerate(evolution.get('rounds', []), 1):
        feedback = item.get('feedback') or {}
        native = item.get('result') or {}
        measurements = feedback.get('measurements') or {}
        construction = phase(native.get('construction') or feedback.get('construction'))
        evaluation = phase(feedback.get('evaluation'))
        updates = []
        for update in feedback.get('updates', []):
            updates.append(dict(kind=clean(update.get('kind')), description=clean(update.get('content')), artifacts=[Path(p).name for p in update.get('artifacts', [])]))
        errors = [clean(e.get('reason')) for e in feedback.get('errors', [])]
        candidate = measurements.get('candidate') or {}
        for failure in candidate.get('failures', []):
            errors.append(clean(failure.get('reason')))
            errors += [clean(stage.get('error') or stage.get('reason')) for stage in failure.get('stages', [])]
        workflow = feedback.get('workflow') or {}
        action = workflow.get('next_action') or {}
        if not errors and action.get('reason'):
            errors.append(clean(action['reason']))
        accepted = bool(item.get('accepted'))
        pending = bool(native.get('version')) and not native.get('evaluation') and not errors
        round_root = Path(item.get('workspace', ''))
        # Archived paths are remapped by relative directory names, never traversed.
        available = root / 'evidence' / 'evolution' / round_root.name
        if not available.exists():
            available = root / 'evolution' / round_root.name
        traces = list(available.glob('session/agent/runs/*/stdout.bin'))
        traces += list(available.glob('session/evaluation-case/benchmark/*/*/session/agents/*/stdout.bin'))
        rounds.append(dict(index=i, status='accepted' if accepted else 'evaluating' if pending else 'rejected', version=native.get('version'), changes=updates, reasons=errors, construction=construction, evaluation=evaluation, correct=candidate.get('correct'), baseline_correct=(measurements.get('baseline') or {}).get('correct'), candidate_seconds=number(candidate.get('seconds')), candidate_tokens=number(candidate.get('total_tokens')), activity=activity(traces)))
    for name in ('construction', 'evaluation'):
        values = [r[name]['seconds'] for r in rounds]
        if phases[name]['seconds'] is None and values and all(v is not None for v in values):
            phases[name]['seconds'] = sum(values)
        elif phases[name]['seconds'] is None and values and name == 'evaluation' and status.get('status') == 'completed':
            # A round with no evaluation has no evaluation elapsed time.
            phases[name]['seconds'] = sum(v for v in values if v is not None)
    live = status.get('status') == 'running'
    if live:
        patterns = dict(construction='evolution/compare-improve*/session/agent/runs/*/stdout.bin', evaluation='evolution/compare-improve*/session/evaluation-case/benchmark/*/*/session/agents/*/stdout.bin')
        for name, pattern in patterns.items():
            traces = list(root.glob(pattern))
            totals = collections.Counter()
            observed = False
            for path in traces:
                with path.open(errors='replace') as stream:
                    for line in stream:
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if event.get('type') == 'status' and event.get('phase') == 'step_end' and isinstance(event.get('usage'), dict):
                            u = event['usage'];observed = True
                            for src, dest in [('inputTokens', 'input_tokens'), ('outputTokens', 'output_tokens'), ('totalTokens', 'total_tokens')]:
                                value = number(u.get(src))
                                if value is not None:
                                    totals[dest] += value
            if observed:
                phases[name]['usage'] = dict(usage(totals), cached_input_tokens=None)
        phase_name = 'evaluation' if list(root.glob('evolution/compare-improve*/session/evaluation-case/benchmark/*/*/session/agents/*/stdout.bin')) else 'construction'
    else:
        phase_name = 'finished'
    launch = read('launch.json')
    elapsed = number(status.get('spent_seconds'))
    if live and launch.get('started_at'):
        try:
            elapsed = max(0, (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(launch['started_at'])).total_seconds())
        except ValueError:
            pass
    live_traces = list(root.glob('evolution/compare-improve*/session/**/stdout.bin')) if live else []
    live_activity = activity(live_traces)
    if live_traces:
        phase_name = 'evaluation' if 'evaluation-case' in str(max(live_traces, key=lambda p: p.stat().st_mtime)) else 'construction'
    postreview = audit.get('postreview') or {}
    notes = []
    if postreview:
        notes.append('维护侧独立复核已有记录；不计作模型自主修复。')
    if audit.get('maintenance'):
        notes.append('维护修复与模型候选分别归档。')
    return dict(id=root.name, model=clean(status.get('model') or experiment.get('model')), commit=experiment.get('frozen_commit') or audit.get('frozen_commit'), status=status.get('status'), phase=phase_name, stop_reason=clean(result.get('reason') or status.get('reason')), accepted=result.get('accepted', audit.get('accepted', 0)), active_version=result.get('active_version', audit.get('active_version', 'empty')), budget_seconds=number(status.get('budget_seconds')), elapsed_seconds=elapsed, live_activity=live_activity, baseline=dict(correct=baseline.get('correct'), reused=bool(baseline.get('reused')), historical=phase(baseline)), phases=phases, rounds=rounds, maintenance_notes=notes, scope='留存真实模型采集 · 开发集反复评测', evidence=dict(archive_sha256=(read('retrieval.json').get('sha256') or (audit.get('archive') or {}).get('sha256'))))


def export(monitor_path, output, refresh=False):
    monitor_path = Path(monitor_path)
    monitor = json.loads(monitor_path.read_text())
    latest = monitor['latest']
    folders = {Path(item['local']).resolve() for item in [latest, monitor.get('previous', {}), *monitor.get('history', [])] if item.get('local')}
    folders.update(p.parent.resolve() for p in (monitor_path.parent / 'runs').glob('*/REPORT.md'))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    current = None
    if refresh:
        ssh = monitor['ssh']
        program = '\n'.join(['import json,collections,math,re,datetime', 'from pathlib import Path', inspect.getsource(summarize), 'print(json.dumps(summarize(' + repr(latest['remote']) + '),ensure_ascii=False,allow_nan=False))'])
        args = ['ssh', '-i', str(Path(ssh['key']).expanduser()), '-o', 'IdentitiesOnly=yes', '-o', 'ClearAllForwardings=yes', '-o', 'ConnectTimeout=12', ssh['alias'], 'python3', '-']
        process = subprocess.run(args, input=program, text=True, capture_output=True, timeout=45, check=True)
        current = json.loads(process.stdout)
        if not current:
            raise ValueError('current remote run has no status receipt')
        current['observed_at'] = now
        cache = Path(latest['local']) / 'dashboard.json'
        cache.write_text(json.dumps(current, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    runs = []
    for folder in sorted(folders, key=lambda p: p.name.rsplit('-', 1)[-1], reverse=True):
        if current and folder == Path(latest['local']).resolve():
            item = current
        elif (folder / 'dashboard.json').exists():
            item = json.loads((folder / 'dashboard.json').read_text())
        else:
            item = summarize(folder)
        if item:
            record = next((record for record in monitor.get('history', []) if Path(record['local']).name == item['id']), {})
            if record.get('acceptance_caveat'):
                item['acceptance_note'] = '历史接受备注：' + record['acceptance_caveat']
            runs.append(item)
    packet = dict(schema_version=1, generated_at=now, observed_at=(current or {}).get('observed_at') or (monitor.get('maintenance', {}).get('last_check') or {}).get('at'), current_run=Path(latest['local']).name, runs=runs)
    target = Path(output)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)
    return packet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monitor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refresh', action='store_true')
    options = parser.parse_args()
    snapshot = export(options.monitor, options.output, options.refresh)
    print(f'Exported {len(snapshot["runs"])} runs; current={snapshot["current_run"]}')
