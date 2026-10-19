#!/usr/bin/env python3
"""Check selected runtime imports and interfaces in the executing environment.

inspect_runtime(spec) runs inside a native worker. probe_runtime(spec,
executable=None, timeout=60) performs the same check in a fresh interpreter and
returns its report plus a bounded diagnostic log. The caller selects the worker
entrypoint and any device operations through task-owned check functions.

Spec: imports=[module,...] in initialization order; symbols=[module:attribute,...];
checks=[module:zero_argument_callable,...] returning JSON values;
distributions=[installed_distribution,...]; python_paths=[absolute_path,...].
Module records contain actual __file__/__version__; distribution metadata is
recorded separately. Python paths apply only during inspection.

CLI: --spec SPEC.json --output NEW.json [--python EXECUTABLE] [--timeout 60]
Resolve relative python_paths against SPEC.json. Exit 0=completed, 1=failed or
timed out, 2=CLI/output error. Requires only the Python standard library.
"""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
import traceback


def _symbol(name):
    module, separator, attribute = name.partition(':')
    if not separator or not module or not attribute:
        raise ValueError('use module:attribute for symbols and checks')
    value = importlib.import_module(module)
    for part in attribute.split('.'):
        value = getattr(value, part)
    return value


def inspect_runtime(spec):
    """Initialize imports in order, resolve interfaces and run selected checks."""
    report = dict(status='completed', executable=sys.executable, python=sys.version,
                  architecture=platform.machine(), host=platform.node(), pid=os.getpid(),
                  modules={}, symbols=[], checks={}, distributions={})
    started = time.monotonic()
    original_paths = sys.path.copy()
    try:
        paths = spec.get('python_paths', [])
        if any(not Path(path).is_absolute() for path in paths):
            raise ValueError('python_paths must be absolute')
        sys.path[:0] = [str(path) for path in paths]
        for name in spec.get('imports', []):
            module = importlib.import_module(name)
            report['modules'][name] = dict(path=getattr(module, '__file__', None),
                                           version=getattr(module, '__version__', None))
        for name in spec.get('symbols', []):
            _symbol(name)
            report['symbols'].append(name)
        for name in spec.get('checks', []):
            result = _symbol(name)()
            json.dumps(result, allow_nan=False)
            report['checks'][name] = result
        for name in spec.get('distributions', []):
            distribution = metadata.distribution(name)
            report['distributions'][name] = dict(version=distribution.version,
                                                 location=str(distribution.locate_file('')))
    except Exception as exc:
        report.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc),
                                                  traceback=traceback.format_exc()))
    finally:
        sys.path[:] = original_paths
    report['elapsed_seconds'] = time.monotonic() - started
    return report


def probe_runtime(spec, executable=None, timeout=60):
    """Run a bounded fresh-process check using the selected interpreter."""
    if not isinstance(timeout, (int, float)) or not 0 < timeout < float('inf'):
        raise ValueError('timeout must be finite and positive')
    payload = json.dumps(spec, allow_nan=False)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='precision-runtime-') as directory:
        result_path = Path(directory) / 'result.json'
        with (Path(directory) / 'worker.log').open('w+b') as log:
            process = subprocess.Popen(
                [str(executable or sys.executable), '-B', str(Path(__file__).resolve()),
                 '--worker', str(result_path)], stdin=subprocess.PIPE, stdout=log,
                stderr=subprocess.STDOUT, text=True, start_new_session=os.name == 'posix')
            try:
                process.communicate(payload, timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.communicate()
                report = dict(status='timeout', executable=str(executable or sys.executable),
                              elapsed_seconds=time.monotonic() - started)
            else:
                if result_path.exists():
                    report = json.loads(result_path.read_text())
                else:
                    report = dict(status='failed', executable=str(executable or sys.executable),
                                  error=dict(type='WorkerExit', message='worker produced no report'))
                if process.returncode and report['status'] == 'completed':
                    report.update(status='failed', error=dict(type='WorkerExit', message='worker exit after checks'))
            log.seek(0, 2)
            size = log.tell()
            log.seek(max(0, size - 8192))
            report.update(returncode=process.returncode, diagnostics=log.read().decode('utf-8', errors='replace'),
                          diagnostic_bytes=size)
    return report


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        report = inspect_runtime(json.load(sys.stdin))
        Path(sys.argv[2]).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        return 0 if report['status'] == 'completed' else 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python')
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        spec = json.loads(args.spec.read_text())
        spec['python_paths'] = [str((args.spec.resolve().parent / path).resolve())
                                for path in spec.get('python_paths', [])]
        result = probe_runtime(spec, args.python, args.timeout)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (ValueError, OSError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
