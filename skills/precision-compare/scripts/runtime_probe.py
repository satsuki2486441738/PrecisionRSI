#!/usr/bin/env python3
"""Check selected runtime interfaces and file-backed resource identities.
Call: probe_runtime(spec, executable=None, timeout=60); inspect_runtime(spec); runtime_identity(report); check_identity(manifest_path)
Input: Ordered runtime checks or a file-backed identity manifest.
Output: Runtime identities/diagnostics or compact file-check status and counts.
Requires: Standard library; native dependencies in the selected interpreter; device execution checks are task-owned.

Details:
check_identity(manifest_path) reads a file-backed {files:[...]} manifest and returns
{status, manifest_sha256, checked_files, checks:{sha256:N,metadata:N}}, or a failed
receipt with bounded error details. Each entry has absolute path and mode:
sha256 (default) requires sha256; metadata requires bytes and mtime_ns, with
optional inode/ctime_ns. Metadata checks reuse a trusted earlier content identity;
they do not rehash file contents. SHA256 entries may also declare bytes. Files must
stay unchanged during their selected check. The receipt omits the file inventory;
require status=completed before using the resource. No manifest code is executed.

inspect_runtime(spec) runs inside a native worker. probe_runtime(spec,
executable=None, timeout=60) performs the same check in a fresh interpreter and
returns its report plus a bounded diagnostic log. The caller selects the worker
entrypoint and any device operations through task-owned check functions.

Spec: imports=[module,...] in initialization order; symbols=[module:attribute,...];
checks=[module:zero_argument_callable,...] returning JSON values;
distributions=[installed_distribution,...]; python_paths=[absolute_path,...].
Module records contain actual __file__/__version__; distribution metadata is
recorded separately. Reports contain plain JSON values, including in native workers.
Python paths apply only during inspection.

runtime_identity(report, checks=()) returns a stable identity from a successful
report: interpreter, host/architecture, imported modules, symbols, distributions
and selected source fingerprints, plus explicitly selected check results. PID,
elapsed time and diagnostics stay in the full report. This supports live probes
whose unchanged identity should permit downstream workflow reuse.

Optional sources=[{"module":"already.imported.module"}, {"path":"/absolute/file"}].
After checks, fingerprint each selected regular file with resolved path, SHA256
and byte count; module entries also retain their name and use the loaded module's
__file__ without importing it. Unavailable or changing files fail inspection,
retaining earlier records. These are on-disk file identities; task checks own
execution/dispatch evidence. Omit sources to keep the existing report shape.

CLI: --spec SPEC.json --output NEW.json [--python EXECUTABLE] [--timeout 60]
Resolve relative python_paths and sources path entries against SPEC.json. Exit 0=completed, 1=failed or
timed out, 2=CLI/output error. Requires only the Python standard library.
"""
import argparse
import hashlib
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


def _file_stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _source_fingerprint(selection):
    if not isinstance(selection, dict) or set(selection) not in ({'module'}, {'path'}):
        raise ValueError('each sources entry needs exactly one module or path')
    if 'module' in selection:
        name = selection['module']
        if not isinstance(name, str) or not name or name not in sys.modules:
            raise ValueError('source module must already be imported')
        value = getattr(sys.modules[name], '__file__', None)
        if value is None:
            raise ValueError(f'source module has no file: {name}')
    else:
        value = selection['path']
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError('source path must be an absolute file path')
    path = Path(value).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f'source is not a regular file: {path}')
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            size += len(block)
        after = os.fstat(stream.fileno())
    if (size != before.st_size or _file_stamp(before) != _file_stamp(after)
            or _file_stamp(before) != _file_stamp(path.stat())):
        raise RuntimeError(f'source file changed during fingerprinting: {path}')
    record = dict(path=str(path), sha256=digest.hexdigest(), bytes=size)
    if 'module' in selection:
        record['module'] = name
    return record


def check_identity(manifest_path):
    """Check selected file identities from a manifest; return counts without its contents."""
    report = dict(status='completed', checked_files=0, checks=dict(sha256=0, metadata=0))
    try:
        data = Path(manifest_path).read_bytes()
        report['manifest_sha256'] = hashlib.sha256(data).hexdigest()
        spec = json.loads(data)
        files = spec.get('files') if isinstance(spec, dict) else None
        if not isinstance(files, list) or not files:
            raise ValueError('nonempty files list required')
        for entry in files:
            if not isinstance(entry, dict) or not isinstance(entry.get('path'), str) or not Path(entry['path']).is_absolute():
                raise ValueError('each identity needs an absolute file path')
            mode = entry.get('mode', 'sha256')
            if mode not in ('sha256', 'metadata'):
                raise ValueError('identity mode must be sha256 or metadata')
            if mode == 'sha256':
                expected = entry.get('sha256')
                if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
                    raise ValueError('SHA256 identity needs a lowercase hexadecimal digest')
            else:
                if any(type(entry.get(k)) is not int or entry[k] < 0 for k in ('bytes', 'mtime_ns')):
                    raise ValueError('metadata identity needs bytes and mtime_ns')
            if any(k in entry and (type(entry[k]) is not int or entry[k] < 0) for k in ('bytes', 'inode', 'ctime_ns')):
                raise ValueError('identity metadata must be nonnegative integers')
        for entry in files:
            mode = entry.get('mode', 'sha256')
            path = Path(entry['path']).resolve(strict=True)
            if not path.is_file():
                raise ValueError('identity path must be a regular file: ' + str(path))
            before = path.stat()
            if mode == 'sha256':
                actual = _source_fingerprint({'path': str(path)})
                if actual['sha256'] != entry['sha256']:
                    raise ValueError('SHA256 mismatch: ' + str(path))
            values = dict(bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                          inode=before.st_ino, ctime_ns=before.st_ctime_ns)
            for key in ('bytes', 'mtime_ns', 'inode', 'ctime_ns'):
                if key in entry and entry[key] != values[key]:
                    raise ValueError(key + ' mismatch: ' + str(path))
            if _file_stamp(before) != _file_stamp(path.stat()):
                raise RuntimeError('identity file changed during check: ' + str(path))
            report['checks'][mode] += 1
            report['checked_files'] += 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        report.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc)[-2048:]))
    return report


def inspect_runtime(spec):
    """Initialize imports in order, resolve interfaces and run selected checks."""
    report = dict(status='completed', executable=sys.executable, python=sys.version,
                  architecture=platform.machine(), host=platform.node(), pid=os.getpid(),
                  modules={}, symbols=[], checks={}, distributions={})
    started = time.monotonic()
    original_paths = sys.path.copy()
    try:
        if 'sources' in spec:
            report['sources'] = []
        paths = spec.get('python_paths', [])
        if any(not Path(path).is_absolute() for path in paths):
            raise ValueError('python_paths must be absolute')
        sys.path[:0] = [str(path) for path in paths]
        for name in spec.get('imports', []):
            module = importlib.import_module(name)
            report['modules'][name] = {key: None if value is None else str(value)
                                      for key, value in (('path', getattr(module, '__file__', None)),
                                                         ('version', getattr(module, '__version__', None)))}
        for name in spec.get('symbols', []):
            _symbol(name)
            report['symbols'].append(name)
        for name in spec.get('checks', []):
            result = _symbol(name)()
            report['checks'][name] = json.loads(json.dumps(result, allow_nan=False))
        for name in spec.get('distributions', []):
            distribution = metadata.distribution(name)
            report['distributions'][name] = dict(version=distribution.version,
                                                 location=str(distribution.locate_file('')))
        selections = spec.get('sources', [])
        if not isinstance(selections, list):
            raise ValueError('sources must be a list')
        for selection in selections:
            report['sources'].append(_source_fingerprint(selection))
    except Exception as exc:
        report.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc),
                                                  traceback=traceback.format_exc()))
    finally:
        sys.path[:] = original_paths
    report['elapsed_seconds'] = time.monotonic() - started
    return json.loads(json.dumps(report, allow_nan=False))


def runtime_identity(report, checks=()):
    """Select reproducible runtime identity fields from a completed native probe."""
    if not isinstance(report, dict) or report.get('status') != 'completed':
        raise ValueError('runtime identity requires a completed probe')
    if (not isinstance(checks, (list, tuple)) or any(not isinstance(k, str) for k in checks)
            or len(set(checks)) != len(checks)):
        raise ValueError('checks must contain unique check names')
    fields = ('executable', 'python', 'architecture', 'host', 'modules', 'symbols',
              'distributions', 'sources')
    result = {key: report[key] for key in fields if key in report}
    if checks:
        if any(key not in report.get('checks', {}) for key in checks):
            raise ValueError('selected runtime check is unavailable')
        result['checks'] = {key: report['checks'][key] for key in checks}
    return json.loads(json.dumps(result, allow_nan=False))


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
                stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                env=dict(os.environ, PYTHONIOENCODING='utf-8'),
                start_new_session=os.name == 'posix')
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
                    report = json.loads(result_path.read_text(encoding='utf-8'))
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
        Path(sys.argv[2]).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return 0 if report['status'] == 'completed' else 1
    parser = argparse.ArgumentParser(description=__doc__.partition('\n\nDetails:\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python')
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        spec = json.loads(args.spec.read_text(encoding='utf-8'))
        spec['python_paths'] = [str((args.spec.resolve().parent / path).resolve())
                                for path in spec.get('python_paths', [])]
        if isinstance(spec.get('sources'), list):
            for selection in spec['sources']:
                if (isinstance(selection, dict) and isinstance(selection.get('path'), str)
                        and selection['path']):
                    selection['path'] = str((args.spec.resolve().parent / selection['path']).resolve())
        result = probe_runtime(spec, args.python, args.timeout)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (ValueError, OSError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
