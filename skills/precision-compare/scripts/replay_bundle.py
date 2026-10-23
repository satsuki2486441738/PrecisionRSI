#!/usr/bin/env python3
"""Replay retained artifacts with installed or explicitly selected skill tools.
Call: replay(bundle, output_root); CLI: python scripts/replay_bundle.py --bundle ZIP --output-root ROOT
Input: Indexed bundle containing REPLAY.json and retained comparison inputs.
Output: Fresh UTC run, per-case reports and replay checks; CLI exit 0 when all checks match.
Requires: Standard library; comparator dependencies at execution; verified in-bundle references; new outputs; default expansion cap 16 GiB.

Details:
API: replay(bundle, output_root=Path('runs'), scripts=None, index='EXPORT.json',
            *, max_expanded_bytes=DEFAULT_MAX_EXPANDED_BYTES)
returns (new_run_path, result). evidence_bundle.extract verifies the ZIP index,
hashes, paths and expansion bound before extraction. Select index='FILES.json'
for a supported FILES index; an index is an archive member, not an external file.

REPLAY.json contains cases:[{name,operation,inputs,expected}]. Names are unique
nonempty strings. Operations are validate (contract/manifest), compare_tensors,
compare_distributions or compare_target_scores (contract/side_a/side_b).
Tensor cases may set mode=metrics|nonfinite; other operations have no mode. Inputs
are relative file paths inside the bundle. Explicit side order is preserved.
Consumed rank environment/status and array references must also be relative
and resolve to retained files inside inputs. Parent references such as ../arrays
are allowed when contained; provenance strings are not loaded-file references.
Expected is a nonempty object selecting nested report fields; lists match in
order and length. Float assertions use rel_tol=abs_tol=1e-12; other scalars
match exactly including their JSON type. Duplicate keys and nonfinite JSON
numbers are rejected. These assertions are replay checks, not model thresholds.

Tools run through their existing CLIs from scripts (default: this script's
directory). No Python code from the bundle is imported or executed. Outputs:
fresh compare-replay-UTC/inputs, case-NN.json reports and replay.json. CLI:
--bundle ZIP [--output-root runs] [--scripts DIR] [--index EXPORT.json]
[--max-expanded-bytes N]. The default expansion bound is 16 GiB.
Exit 0=all assertions match, 1=mismatch or invalid inputs/tool execution.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

from evidence_bundle import DEFAULT_MAX_EXPANDED_BYTES, extract


OPERATIONS = {
    'validate': ('validate_artifacts.py', ('contract', 'manifest')),
    'compare_tensors': ('compare_tensors.py', ('contract', 'side_a', 'side_b')),
    'compare_distributions': ('compare_distributions.py', ('contract', 'side_a', 'side_b')),
    'compare_target_scores': ('compare_target_scores.py', ('contract', 'side_a', 'side_b')),
}
TOLERANCE = 1e-12


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_json(path):
    """Read unique-key JSON, rejecting NaN, Infinity and overflowing floats."""
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate JSON key: ' + key)
            value[key] = item
        return value

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('nonfinite JSON number: ' + value)
        return number

    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=pairs, parse_float=finite,
                         parse_constant=finite)


def safe_path(root, name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('invalid replay input path: ' + repr(name))
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != name or name == '.':
        raise ValueError('invalid replay input path: ' + repr(name))
    root = Path(root).resolve()
    result = root.joinpath(*path.parts).resolve()
    if not result.is_relative_to(root):
        raise ValueError('replay input path leaves input directory')
    return result


def check_subset(actual, expected, label='result'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(label + ': expected object')
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(label + '.' + key + ': missing')
            check_subset(actual[key], value, label + '.' + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(label + ': list length/type differs')
        for index, (observed, wanted) in enumerate(zip(actual, expected)):
            check_subset(observed, wanted, f'{label}[{index}]')
    elif isinstance(expected, float):
        if (not math.isfinite(expected) or type(actual) not in (int, float)
                or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE)):
            raise AssertionError(f'{label}: {actual!r} != {expected!r}')
    elif type(actual) is not type(expected) or actual != expected:
        raise AssertionError(f'{label}: {actual!r} != {expected!r}')


def replay_cases(inputs):
    document = load_json(inputs / 'REPLAY.json')
    if not isinstance(document, dict):
        raise ValueError('REPLAY.json must be an object containing cases')
    cases = document.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('nonempty replay cases list required')
    names, normalized = set(), []
    for case in cases:
        required_fields = {'name', 'operation', 'inputs', 'expected'}
        if (not isinstance(case, dict) or not required_fields <= set(case)
                or set(case) - required_fields - {'mode'}):
            raise ValueError('replay case requires name, operation, inputs and expected; optional tensor mode')
        # Decode archived cases into the current tensor interface in memory.
        if case['operation'] == 'inspect_nonfinite':
            if 'mode' in case:
                raise ValueError('archived nonfinite operation cannot include mode')
            case = dict(case, operation='compare_tensors', mode='nonfinite')
        name = case['name']
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('unique nonempty replay case names required')
        names.add(name)
        operation = case['operation']
        if not isinstance(operation, str) or operation not in OPERATIONS:
            raise ValueError('unsupported replay operation: ' + repr(operation))
        if 'mode' in case and (operation != 'compare_tensors' or case['mode'] not in ('metrics', 'nonfinite')):
            raise ValueError('mode must be metrics or nonfinite for compare_tensors')
        required = OPERATIONS[operation][1]
        if not isinstance(case['inputs'], dict) or set(case['inputs']) != set(required):
            raise ValueError(operation + ' inputs must contain exactly ' + ', '.join(required))
        for key in required:
            path = safe_path(inputs, case['inputs'][key])
            if not path.is_file():
                raise ValueError('replay input file missing: ' + case['inputs'][key])
        if not isinstance(case['expected'], dict) or not case['expected']:
            raise ValueError('replay expected must be a nonempty result-subset object')
        normalized.append(case)
    return normalized


def validate_references(cases, inputs):
    """Check only file references consumed by supported tool schemas, before runs."""
    inputs = inputs.resolve()
    documents = {}

    def read(path):
        if path not in documents:
            documents[path] = load_json(path)
        return documents[path]

    def reference(document, name):
        if not isinstance(name, str) or not name or '\\' in name or ':' in name:
            raise ValueError('invalid retained artifact reference: ' + repr(name))
        path = PurePosixPath(name)
        if path.is_absolute() or path.as_posix() != name or name == '.':
            raise ValueError('invalid retained artifact reference: ' + repr(name))
        resolved = document.parent.joinpath(*path.parts).resolve()
        if not resolved.is_relative_to(inputs):
            raise ValueError('artifact reference leaves retained inputs: ' + repr(name))
        if not resolved.is_file():
            raise ValueError('referenced artifact not retained: ' + repr(name))

    def records(value):
        # Malformed semantic fields remain the comparator's diagnostic concern.
        return (row for row in value if isinstance(row, dict)) if isinstance(value, list) else ()

    def array(document, descriptor):
        if isinstance(descriptor, dict) and 'path' in descriptor:
            reference(document, descriptor['path'])

    for case in cases:
        operation = case['operation']
        if operation == 'validate':
            contract = read(safe_path(inputs, case['inputs']['contract']))
            kind = contract.get('kind') if isinstance(contract, dict) else None
            keys = ('manifest',)
        else:
            kind = {'compare_tensors': 'tensors', 'compare_distributions': 'distributions',
                    'compare_target_scores': 'target-scores'}[operation]
            keys = ('side_a', 'side_b')
        for key in keys:
            document = safe_path(inputs, case['inputs'][key])
            manifest = read(document)
            if not isinstance(manifest, dict):
                continue
            for rank in records(manifest.get('ranks')):
                for field in ('environment', 'status'):
                    if field in rank:
                        reference(document, rank[field])
            if kind == 'tensors':
                for tensor in records(manifest.get('tensors')):
                    for shard in records(tensor.get('shards')):
                        for field in ('value', 'before', 'after'):
                            array(document, shard.get(field))
            elif kind == 'distributions':
                for position in records(manifest.get('positions')):
                    array(document, position.get('scores'))


def run_case(case, inputs, scripts, output):
    tool, keys = OPERATIONS[case['operation']]
    tool_path = scripts / tool
    if not tool_path.is_file():
        raise ValueError('selected skill tool missing: ' + str(tool_path))
    command = [sys.executable, '-B', str(tool_path)]
    for key in keys:
        command.extend(['--' + key.replace('_', '-'), str(safe_path(inputs, case['inputs'][key]))])
    if 'mode' in case:
        command.extend(['--mode', case['mode']])
    command.extend(['--output', str(output)])
    completed = subprocess.run(command, cwd=scripts, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', check=False)
    # Exit 1 with a report is a valid unavailable comparison/validation outcome.
    if completed.returncode not in (0, 1) or not output.is_file():
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f'{tool} failed (exit {completed.returncode}): {detail}')
    report = load_json(output)
    if not isinstance(report, dict):
        raise ValueError('skill tool report must be an object: ' + tool)
    return report


def replay(bundle, output_root=Path('runs'), scripts=None, index='EXPORT.json',
           *, max_expanded_bytes=DEFAULT_MAX_EXPANDED_BYTES):
    bundle = Path(bundle).resolve()
    scripts = Path(scripts if scripts is not None else Path(__file__).resolve().parent).resolve()
    if not scripts.is_dir():
        raise ValueError('skill scripts directory missing: ' + str(scripts))
    root = Path(output_root).resolve() / datetime.now(timezone.utc).strftime('compare-replay-%Y%m%dT%H%M%SZ')
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / 'inputs'
    receipt = extract(bundle, inputs, index=index, max_expanded_bytes=max_expanded_bytes)
    cases = replay_cases(inputs)
    validate_references(cases, inputs)
    results = []
    for number, case in enumerate(cases):
        filename = f'case-{number:02d}.json'
        report = run_case(case, inputs, scripts, root / filename)
        result = dict(name=case['name'], report=filename, assertion='matched')
        try:
            check_subset(report, case['expected'])
        except AssertionError as exc:
            result.update(assertion='failed', reason=str(exc))
        results.append(result)
    result = dict(bundle_sha256=receipt['archive_sha256'], index=index,
                  max_expanded_bytes=max_expanded_bytes,
                  scripts=str(scripts), cases=results,
                  matched=all(case['assertion'] == 'matched' for case in results))
    write(root / 'replay.json', result)
    return root, result


def main():
    parser = argparse.ArgumentParser(description=__doc__.partition('\n\nDetails:\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('runs'))
    parser.add_argument('--scripts', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--index', default='EXPORT.json')
    parser.add_argument('--max-expanded-bytes', type=int, default=DEFAULT_MAX_EXPANDED_BYTES)
    args = parser.parse_args()
    try:
        root, result = replay(args.bundle, args.output_root, args.scripts, args.index,
                              max_expanded_bytes=args.max_expanded_bytes)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, zipfile.BadZipFile) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(dict(run=str(root), matched=result['matched'], cases=len(result['cases']))))
    return 0 if result['matched'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
