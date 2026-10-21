#!/usr/bin/env python3
"""Copy only the skill package and run its CPU example from an unrelated cwd.

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


def main():
    source = Path(__file__).resolve().parents[1] / 'skills/precision-skill'
    with tempfile.TemporaryDirectory(prefix='precision-install-') as directory:
        root = Path(directory)
        installed = root / 'skills/precision-skill'
        shutil.copytree(source, installed, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        cwd = root / 'work'
        cwd.mkdir()
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
        result = subprocess.run([sys.executable, '-I', '-B', str(installed / 'examples/compare_gradients.py'),
                                 '--output-root', str(cwd)], cwd=cwd, env=env, capture_output=True,
                                text=True, check=True, timeout=120)
        output = json.loads(result.stdout)
        artifacts = Path(output['artifacts'])
        if artifacts.parent != cwd:
            raise ValueError('example wrote outside selected output root')
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
        print('Clean skill installation: CPU FP32/BF16 captures validated; compared 12 gradient values.')


if __name__ == '__main__':
    main()
