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
        if report['status'] != 'no_thresholds' or report['global']['count'] != 12:
            raise ValueError(f'incomplete gradient comparison: {report}')
        for side in ('fp32', 'bf16'):
            validation = json.loads((artifacts / side / 'validation.json').read_text())
            environment = json.loads((artifacts / side / 'environment.json').read_text())
            if validation['status'] != 'validated' or environment['actual']['numerics']['device'] != 'cpu':
                raise ValueError(f'invalid CPU capture for {side}')
        for key in ('side_a_norm', 'side_b_norm', 'error_norm', 'rmse', 'cosine', 'max_abs'):
            metric = report['global'][key]
            if metric['status'] != 'no_thresholds' or not math.isfinite(metric['value']):
                raise ValueError(f'invalid metric {key}: {metric}')
        if not all(math.isfinite(value) for value in output['losses'].values()):
            raise ValueError('nonfinite native loss')
        execution = subprocess.run(
            [sys.executable, '-I', '-B', str(installed / 'examples/workflow_demo.py'),
             '--output-root', str(cwd)], cwd=cwd, env=env, capture_output=True,
            text=True, check=True, timeout=120)
        workflow = json.loads(execution.stdout)
        validate_artifact_directory(workflow['case'], cwd)
        if (not workflow['routed_to_repair'] or not workflow['preparation_reused']
                or not workflow['resource_discovered'] or workflow['replay_native_commands'] != 0
                or workflow['before_max_abs'] <= 1 or workflow['after_max_abs'] > 1e-6):
            raise ValueError(f'incomplete workflow: {workflow}')
        print(f'Clean installation of {len(folders)} skills: links valid; '
              'CPU FP32/BF16 captures validated; compared 12 gradient values; '
              'native repair workflow verified with zero-command replay.')


if __name__ == '__main__':
    main()
