"""Run a pinned upstream regression fixture and compare its numerical output."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

import ascend_cases
import mindspeed_cases
import mindspeed_llm_cases
import sglang_cases
import vllm_cases


COMPARATOR = Path(__file__).resolve().parents[2] / 'skills/precision-compare/scripts/array_compare.py'
CASES = {
    **ascend_cases.CASES,
    **mindspeed_cases.CASES,
    **mindspeed_llm_cases.CASES,
    **sglang_cases.CASES,
    **vllm_cases.CASES,
}


def case_metadata(case_id):
    case = CASES[case_id]
    return {'id': case_id, **{key: value for key, value in case.items() if key != 'run'}}


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def package_versions():
    versions = {}
    for package in ('numpy', 'torch', 'torch-npu', 'triton', 'triton-ascend', 'sglang', 'vllm-ascend'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def run_case(args):
    case = CASES[args.case]
    reference, observed = case['run'](args)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_root.resolve() / f'compare-{args.case}-{args.mode}-{args.variant}-{timestamp}'
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'reference.npz', **reference)
    np.savez_compressed(output / 'observed.npz', **observed)
    write_json(output / 'expected.json', sorted(reference))

    report_path = output / 'report.json'
    result = subprocess.run([
        sys.executable, '-B', str(COMPARATOR),
        output / 'reference.npz', output / 'observed.npz',
        '--expected', output / 'expected.json', '--output', report_path,
        '--mode', 'nonfinite', '--metrics',
    ], check=False)
    if not report_path.exists():
        raise RuntimeError(f'NumericBridge comparison failed, exit {result.returncode}: {output}')
    report = json.loads(report_path.read_text(encoding='utf-8'))

    atol, rtol = case.get('atol', 0), case.get('rtol', 0)
    acceptance = 'incomplete'
    if result.returncode == 0 and report['coverage']['compared_elements']:
        matches = report['exact_equal']
        if (atol or rtol) and not report['dtype_differences'] and not any(report['nonfinite'].values()):
            matches = all(np.allclose(observed[name], array, atol=atol, rtol=rtol)
                          for name, array in reference.items())
        acceptance = 'match' if matches else 'mismatch'
    record = {
        'case': case_metadata(args.case),
        'capture_mode': args.mode,
        'variant': args.variant,
        'upstream_revision': case[f'revision_{args.variant}'] if args.mode != 'cpu-smoke' else None,
        'device': args.device if args.mode == 'native-npu' else 'cpu',
        'runtime_label': args.runtime_label,
        'environment': {'system': platform.system(), 'machine': platform.machine(),
                        'python': platform.python_version(), 'packages': package_versions()},
        'acceptance': {'outcome': acceptance, 'expected': args.expect, 'atol': atol, 'rtol': rtol},
    }
    write_json(output / 'capture.json', record)
    print(json.dumps({
        'output': str(output),
        'acceptance': record['acceptance'],
        'coverage': report['coverage'],
        'unequal_elements': report['unequal_elements'],
        'nonfinite': report['nonfinite'],
        'metrics': report['metrics'],
    }, indent=2, allow_nan=False))
    return int(acceptance == 'incomplete' or args.expect not in (None, acceptance))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='print case IDs, upstream revisions and supported modes as JSON')
    run = commands.add_parser('run', help='execute a fixture and write paired NPZ and JSON artifacts')
    run.add_argument('case', choices=sorted(CASES))
    run.add_argument('--mode', required=True, choices=('source-cpu', 'cpu-smoke', 'native-npu'))
    run.add_argument('--variant', required=True, choices=('before', 'after'))
    run.add_argument('--source', type=Path, help='local upstream Git repository containing the pinned source objects')
    run.add_argument('--device', default='npu:0')
    run.add_argument('--runtime-label', help='build provenance, accelerator model and CANN version for a native run')
    run.add_argument('--output-root', type=Path, default=Path(tempfile.gettempdir()) / 'numericbridge-cases')
    run.add_argument('--expect', choices=('match', 'mismatch'), help='fail if the observed comparison differs')
    args = parser.parse_args()
    if args.command == 'list':
        print(json.dumps([case_metadata(case_id) for case_id in sorted(CASES)], indent=2))
        return 0
    if args.mode not in CASES[args.case]['modes']:
        parser.error(f'{args.case} does not support {args.mode}')
    if args.mode == 'source-cpu' and args.source is None:
        parser.error('source-cpu requires --source')
    if args.mode == 'native-npu' and not args.runtime_label:
        parser.error('native-npu requires --runtime-label with installed build provenance')
    return run_case(args)


if __name__ == '__main__':
    raise SystemExit(main())
