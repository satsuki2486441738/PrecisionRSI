#!/usr/bin/env python3
"""Copy installable skills, validate their links and run the CPU comparison example.

Requires Torch and NumPy in this interpreter. No model downloads or accelerators.
All installation and execution artifacts are temporary.
"""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def install_skills(source, destination):
    """Copy package directories without repository maintenance files or caches."""
    for folder in sorted(source.iterdir()):
        if folder.is_dir() and not folder.name.startswith('.'):
            shutil.copytree(folder, destination / folder.name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))


def validate_artifact_directory(artifacts, output_root):
    """Require the example run directory to belong to the selected output root."""
    artifacts = Path(artifacts).resolve()
    if artifacts.parent != Path(output_root).resolve():
        raise ValueError('example wrote outside selected output root')
    return artifacts

def main():
    import numpy as np
    from validate import validate_links, validate_skills
    source = Path(__file__).resolve().parents[1] / 'skills'
    with tempfile.TemporaryDirectory(prefix='precision-install-') as directory:
        root = Path(directory)
        install_skills(source, root / 'skills')
        folders = validate_skills(root)
        _, errors = validate_links(root, sorted((root / 'skills').rglob('*.md')))
        if errors:
            raise ValueError('\n'.join(errors))
        installed = root / 'skills/precision-compare'
        cwd = root / 'work'
        cwd.mkdir()
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
        result = subprocess.run([sys.executable, '-I', '-B', str(installed / 'examples/compare_gradients.py'),
                                 '--output-root', str(cwd)], cwd=cwd, env=env, capture_output=True,
                                text=True, check=True, timeout=120)
        output = json.loads(result.stdout)
        artifacts = validate_artifact_directory(output['artifacts'], cwd)
        report = json.loads((artifacts / 'report.json').read_text())
        capture = json.loads((artifacts / 'capture.json').read_text())
        if report['status'] != 'completed' or report['coverage']['compared_elements'] != 12:
            raise ValueError(f'incomplete gradient comparison: {report}')
        if set(p.name for p in artifacts.iterdir()) != {'fp32.npz', 'bf16.npz', 'capture.json', 'report.json'}:
            raise ValueError('unexpected native example artifacts')
        common = capture['common']
        if capture['expected'] != ['weight']:
            raise ValueError('incomplete parameter inventory')
        for side, dtype in [('fp32', 'torch.float32'), ('bf16', 'torch.bfloat16')]:
            actual = capture['sides'][side]
            if (actual['numerics']['device'] != 'cpu' or actual['inputs'] != common['inputs']
                    or actual['state'] != common['state'] or actual['precision']['gradient'] != dtype):
                raise ValueError(f'invalid CPU capture for {side}')
        for key in ('side_a_norm', 'side_b_norm', 'error_norm', 'rmse', 'cosine', 'max_abs'):
            metric = report['metrics'][key]
            if metric['status'] != 'no_thresholds' or not math.isfinite(metric['value']):
                raise ValueError(f'invalid metric {key}: {metric}')
        if not all(math.isfinite(value) for value in output['losses'].values()):
            raise ValueError('nonfinite native loss')

        # Maintain manifest validation/parity here, outside the user-facing example.
        def write(path, value):
            path.write_text(json.dumps(value, allow_nan=False) + '\n')
        check_root = cwd / 'manifest-check'
        check_root.mkdir()
        sides = {side: dict(precision=actual['precision']) for side, actual in capture['sides'].items()}
        write(check_root / 'contract.json', dict(schema_version=2, kind='tensors',
              common=common, side_conditions=sides, thresholds=None, stage='gradients',
              expected_ranks={side: [0] for side in sides},
              tensors=[dict(name='weight', shape=[3, 4], present=True)]))
        for side, actual in capture['sides'].items():
            folder = check_root / side
            folder.mkdir()
            with np.load(artifacts / (side + '.npz'), allow_pickle=False) as values:
                if (set(values.files) != set(capture['expected']) or values['weight'].shape != (3, 4)
                        or str(values['weight'].dtype) != actual['precision']['exported_gradient']):
                    raise ValueError('invalid native array inventory, shape or export dtype')
                np.save(folder / 'gradient.npy', values['weight'], allow_pickle=False)
            write(folder / 'environment.json', dict(rank=0, versions=capture['versions'], actual=common | actual))
            write(folder / 'status.json', dict(rank=0, status='completed'))
            write(folder / 'manifest.json', dict(schema_version=2, kind='tensors', common=common,
                  side=side, expected_ranks=[0], stage='gradients',
                  ranks=[dict(rank=0, environment='environment.json', status='status.json')],
                  tensors=[dict(name='weight', shape=[3, 4], present=True,
                                shards=[dict(offset=0, value={'path': 'gradient.npy'})])]))
        command = """import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from compare_tensors import compare
from validate_artifacts import validate
root=Path(sys.argv[2]);contract=root/'contract.json'
for side in ('fp32','bf16'):
    checked=validate(contract,root/side/'manifest.json')
    assert checked['status']=='validated',checked
report=compare(contract,root/'fp32/manifest.json',root/'bf16/manifest.json')
assert report['status']=='no_thresholds',report
(root/'report.json').write_text(json.dumps(report,allow_nan=False))
"""
        subprocess.run([sys.executable, '-I', '-B', '-c', command,
                        str(installed / 'scripts'), str(check_root)], cwd=cwd, env=env,
                       capture_output=True, text=True, check=True, timeout=30)
        manifest_report = json.loads((check_root / 'report.json').read_text())
        for key in ('side_a_norm', 'side_b_norm', 'error_norm', 'rmse', 'cosine', 'max_abs'):
            if report['metrics'][key] != manifest_report['global'][key]:
                raise ValueError('array/manifest metric mismatch: ' + key)
        subprocess.run(
            [sys.executable, '-I', '-B', str(installed / 'scripts/array_compare.py'),
             str(artifacts / 'fp32.npz'), str(artifacts / 'bf16.npz'), '--metrics',
             '--output', str(cwd / 'arrays.json')], cwd=cwd, env=env,
            capture_output=True, text=True, check=True, timeout=30)
        if json.loads((cwd / 'arrays.json').read_text()) != report:
            raise ValueError('array CLI/API comparison mismatch')
        execution = subprocess.run(
            [sys.executable, '-I', '-B', str(installed / 'examples/workflow_demo.py'),
             '--output-root', str(cwd)], cwd=cwd, env=env, capture_output=True,
            text=True, check=True, timeout=120)
        workflow = json.loads(execution.stdout)
        validate_artifact_directory(workflow['case'], cwd)
        if (not workflow['routed_to_repair'] or not workflow['preparation_reused']
                or workflow['replay_native_commands'] != 0
                or workflow['before_max_abs'] <= 1 or workflow['after_max_abs'] > 1e-6):
            raise ValueError(f'incomplete workflow: {workflow}')
        # Resource discovery is a cross-workload interface, separate from the demo.
        catalog_check = """
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from resource_catalog import register,find
case=Path(sys.argv[2])
prepared=json.loads((case/'before.json').read_text())['stages'][0]
runtime=json.loads(Path(prepared['outputs']['runtime']['path']).read_text())
root=Path.cwd()/'resource-check'
root.mkdir()
declaration=root/'environment.json'
declaration.write_bytes((case/'environment.json').read_bytes())
resources=dict(declaration=str(declaration),runtime=prepared['outputs']['runtime']['path'],python=sys.executable)
for index,module in enumerate(runtime['modules'].values()):
    if module['path']:resources['module'+str(index)]=module['path']
entry=root/'entry.json'
entry.write_text(json.dumps(dict(id='native-torch-cpu',kind='environment',
    properties=dict(framework='torch',device='cpu'),paths=resources,
    evidence=str(Path(prepared['run'])/'command.json'))))
catalog=root/'catalog.json'
assert register(catalog,entry)['registered']
assert not register(catalog,entry)['registered']
found=find(catalog,'environment',dict(framework='torch',device='cpu'))
assert len(found)==1 and Path(found[0]['paths']['runtime']['path']).is_file()
declaration.write_text('{}')
assert find(catalog,'environment',dict(framework='torch',device='cpu'))==[]
"""
        subprocess.run([sys.executable, '-I', '-B', '-c', catalog_check,
                        str(installed / 'scripts'), workflow['case']], cwd=cwd, env=env,
                       capture_output=True, text=True, check=True, timeout=30)
        print(f'Clean installation of {len(folders)} skills: links valid; '
              'CPU FP32/BF16 captures validated; array/manifest metrics agree for 12 gradient values; '
              'native repair workflow verified with zero-command replay.')


if __name__ == '__main__':
    main()
