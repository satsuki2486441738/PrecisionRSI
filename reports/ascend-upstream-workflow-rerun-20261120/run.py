"""Run a pinned CPU regression through the NumericBridge workflow engine."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

import mindspeed_case
import vllm_case


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMPARATOR = ROOT / 'skills/precision-compare/scripts/array_compare.py'
CASES = {**mindspeed_case.CASES, **vllm_case.CASES}


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def metadata_for(case):
    return {key: value for key, value in case.items() if key != 'run'}


def readiness(args, output):
    revisions = {variant: CASES[args.case][f'revision_{variant}'] for variant in ('before', 'after')}
    dependencies = CASES[args.case].get('dependencies', {})
    checkouts = [(args.source, revision) for revision in revisions.values()]
    checkouts.extend((args.source / project.rsplit('/', 1)[-1], revision)
                     for project, revision in dependencies.items())
    for checkout, revision in checkouts:
        subprocess.run(['git', '-C', checkout, '-c', 'protocol.allow=never',
                        'cat-file', '-e', f'{revision}^{{commit}}'], check=True)
    write(output / 'readiness.json', dict(status='completed', revisions=revisions, system=platform.system(),
                                         machine=platform.machine(), python=platform.python_version(),
                                         torch=metadata.version('torch'), numpy=np.__version__,
                                         dependencies=dependencies))


def preflight(output):
    import torch

    fixture = torch.tensor([1.0, 2.0], device='cpu', requires_grad=True)
    gradient, = torch.autograd.grad(fixture.square().sum(), fixture)
    write(output / 'preflight.json', dict(device=str(gradient.device),
                                         passed=torch.equal(gradient, torch.tensor([2.0, 4.0]))))


def capture(args, output):
    case = CASES[args.case]
    args.mode, args.device = 'source-cpu', 'cpu'
    reference, observed = case['run'](args)
    np.savez_compressed(output / 'reference.npz', **reference)
    np.savez_compressed(output / 'observed.npz', **observed)
    write(output / 'expected.json', sorted(reference))
    report_path = output / 'comparison.json'
    subprocess.run([sys.executable, '-B', COMPARATOR, output / 'reference.npz', output / 'observed.npz',
                    '--expected', output / 'expected.json', '--output', report_path,
                    '--mode', 'nonfinite', '--metrics'], check=True)
    report = json.loads(report_path.read_text())
    coverage = report['coverage']
    complete = (coverage['compared_arrays'] == len(reference)
                and coverage['compared_elements'] == sum(np.asarray(value).size for value in reference.values())
                and coverage['compared_elements'] > 0)
    outcome = 'match' if report['exact_equal'] else 'mismatch'
    write(output / 'capture.json', {
        'case_id': args.case, 'mode': args.mode, 'variant': args.variant,
        'upstream_revision': case[f'revision_{args.variant}'],
        'coverage_complete': complete,
        'compared_arrays': coverage['compared_arrays'], 'compared_elements': coverage['compared_elements'],
        'unequal_elements': report['unequal_elements'], 'nonfinite': report['nonfinite'],
        'acceptance': {'outcome': outcome, 'expected': 'mismatch' if args.variant == 'before' else 'match',
                       'atol': 0, 'rtol': 0},
        'max_abs': report['metrics']['max_abs']['value'], 'rmse': report['metrics']['rmse']['value'],
    })


def plan(args):
    driver = str(Path(__file__).resolve())
    argv = [sys.executable, '-B', driver, 'step', args.case, '--source', str(args.source), '--output', '{run}']
    inputs = [driver, str(HERE / 'mindspeed_case.py'), str(HERE / 'vllm_case.py')]

    def stage(name, skill, output, args_suffix, predicates, depends_on=()):
        return dict(id=name, skill=skill, argv=argv + args_suffix, inputs=inputs,
                    outputs={'receipt': output}, depends_on=list(depends_on), timeout=35,
                    checks=[dict(output='receipt', pointer=pointer, equals=value) for pointer, value in predicates])

    stages = [
        stage('source-readiness', 'framework-adapt', 'readiness.json', ['--stage', 'readiness'],
              [('/status', 'completed')]),
        stage('cpu-preflight', 'framework-adapt', 'preflight.json', ['--stage', 'preflight'],
              [('/device', 'cpu'), ('/passed', True)], ['source-readiness']),
    ]
    stages[0]['cache'] = False
    for variant, expected in (('before', 'mismatch'), ('after', 'match')):
        side = stage(variant, 'precision-repair', 'capture.json', ['--stage', 'capture', '--variant', variant],
                     [('/coverage_complete', True), ('/acceptance/outcome', expected),
                      ('/nonfinite/side_a', 0), ('/nonfinite/side_b', 0)], ['cpu-preflight'])
        side['inputs'] = [*inputs, *(str(path) for path in sorted(COMPARATOR.parent.glob('*.py')))]
        side['outputs'].update(comparison='comparison.json', reference='reference.npz', observed='observed.npz')
        stages.append(side)
    return dict(schema_version=1, id=args.case, root=str(ROOT), budget_seconds=120,
                workload=metadata_for(CASES[args.case]), stages=stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    run = commands.add_parser('run')
    step = commands.add_parser('step', help=argparse.SUPPRESS)
    for command in (run, step):
        command.add_argument('case', choices=sorted(CASES))
        command.add_argument('--source', type=Path, required=True)
    run.add_argument('--workspace', type=Path, required=True)
    step.add_argument('--stage', required=True, choices=('readiness', 'preflight', 'capture'))
    step.add_argument('--variant', choices=('before', 'after'))
    step.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'list':
        print(json.dumps({key: metadata_for(value) for key, value in CASES.items()}, indent=2))
        return 0
    args.source = args.source.resolve()
    if args.command == 'step':
        if args.stage == 'readiness':
            readiness(args, args.output)
        elif args.stage == 'preflight':
            preflight(args.output)
        else:
            capture(args, args.output)
        return 0
    args.workspace.mkdir(parents=True, exist_ok=True)
    plan_path = args.workspace / 'plan.json'
    document = plan(args)
    if not plan_path.exists():
        write(plan_path, document)
    elif json.loads(plan_path.read_text()) != document:
        parser.error('Use a new workspace after changing the case, source checkout or workflow plan')
    return subprocess.run([sys.executable, '-B', '-m', 'tools.workflow', 'run', '--plan', plan_path.resolve(),
                           '--workspace', args.workspace.resolve()], cwd=ROOT, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
