#!/usr/bin/env python3
"""Record one argv execution in a fresh UTC run directory (standard library only).

CLI: record_run.py --output-root ROOT --side ID --task TASK --timeout SECONDS
                  [--cwd DIR] -- ARGV...
API: record_run(output_root, side, task, timeout, argv, cwd=None) -> (Path, dict).
Python callers may pass argv as a callable(Path) -> literal argv to use the newly
reserved run directory. Preparation errors retain a launch_failed run without
starting a child. CLI argv is always literal.
IDs contain ASCII alphanumeric segments separated by single hyphens. Timeout
must be finite and positive. Invalid inputs and existing run directories fail
before launch; existing runs are never reused. The name is
<side>-<task>-YYYYmmddTHHMMSSZ. Same-second collisions are errors, not retries.

command.json records start/end UTC, literal argv, resolved cwd, timeout, child
PID, status, returncode, raw child_returncode, and monotonic wall_seconds.
stdout.bin/stderr.bin receive bytes directly from the child, without buffering
output in the recorder. No environment snapshot is taken. Argv and child output
are retained as supplied. Execution uses no shell. Status is running, completed,
failed, timed_out, launch_failed, or interrupted. Normal exits retain the child
exit code; signals become 128 + signal; timeout returns 124, launch failure 127,
interruption 130. CLI validation/output errors return 2.

POSIX children start a new session; on timeout/interruption, terminate that
owned process group, wait up to 0.5 seconds for the leader, then kill the group
(including descendants that ignore SIGTERM). Descendants that deliberately
leave the process group are outside this guarantee. Elsewhere only the direct
child is terminated. Ordinary successful completion does not kill background
work. Files and command metadata remain in place on failure.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def _utc_now():
    return datetime.now(timezone.utc)


def _write_record(folder, record):
    temporary = folder / 'command.json.tmp'
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(folder / 'command.json')


def _terminate(process):
    if os.name == 'posix':
        def send(sig):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        send(signal.SIGTERM)
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        # The leader may have exited while a descendant still owns the group.
        send(signal.SIGKILL)
    else:
        process.kill()
    process.wait()


def record_run(output_root, side, task, timeout, argv, cwd=None):
    """Validate, reserve a unique run, and execute literal argv; see module contract."""
    for name, value in [('side', side), ('task', task)]:
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*', value):
            raise ValueError(name + ' must contain alphanumeric segments separated by single hyphens')
    if isinstance(timeout, bool):
        raise ValueError('timeout must be finite and positive')
    try:
        timeout = float(timeout)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('timeout must be finite and positive') from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    def validate_argv(value):
        if (not isinstance(value, (list, tuple)) or not value
                or any(not isinstance(a, str) or '\0' in a for a in value) or not value[0]):
            raise ValueError('argv must be a nonempty sequence of strings without NUL bytes')
        return list(value)
    prepared = None if callable(argv) else validate_argv(argv)
    directory = Path.cwd() if cwd is None else Path(cwd).resolve()
    if not directory.is_dir():
        raise ValueError('cwd must be an existing directory')
    started = _utc_now()
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    folder = root / f'{side}-{task}-{started.strftime("%Y%m%dT%H%M%SZ")}'
    folder.mkdir(exist_ok=False)
    record = dict(start=started.isoformat(), end=None, argv=prepared, cwd=str(directory),
                  timeout=timeout, child_pid=None, returncode=None, child_returncode=None,
                  status='running', wall_seconds=None)
    begin = time.monotonic()
    _write_record(folder, record)
    with (folder / 'stdout.bin').open('xb') as stdout, (folder / 'stderr.bin').open('xb') as stderr:
        try:
            if prepared is None:
                prepared = validate_argv(argv(folder))
                record['argv'] = prepared
                _write_record(folder, record)
            process = subprocess.Popen(prepared, cwd=directory, stdout=stdout, stderr=stderr,
                                       start_new_session=os.name == 'posix')
        except Exception as exc:
            record.update(status='launch_failed', returncode=127,
                          error=dict(type=type(exc).__name__, message=str(exc)))
        else:
            record['child_pid'] = process.pid
            _write_record(folder, record)
            try:
                raw_code = process.wait(timeout=timeout)
                record.update(status='completed' if raw_code == 0 else 'failed',
                              returncode=raw_code if raw_code >= 0 else 128 - raw_code)
            except subprocess.TimeoutExpired:
                _terminate(process)
                record.update(status='timed_out', returncode=124)
            except KeyboardInterrupt:
                _terminate(process)
                record.update(status='interrupted', returncode=130)
            record['child_returncode'] = process.returncode
    record.update(end=_utc_now().isoformat(), wall_seconds=time.monotonic() - begin)
    _write_record(folder, record)
    return folder, record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--side', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--timeout', required=True, type=float)
    parser.add_argument('--cwd')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    # Require the explicit separator so child flags cannot become recorder flags.
    if not args.command or args.command[0] != '--':
        parser.error('supply -- followed by the child argv')
    try:
        folder, record = record_run(args.output_root, args.side, args.task, args.timeout,
                                    args.command[1:], args.cwd)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(dict(run=str(folder), status=record['status'], returncode=record['returncode'])))
    return record['returncode']


if __name__ == '__main__':
    sys.exit(main())
