#!/usr/bin/env python3
"""Generate, validate and compare synthetic gradients; requires NumPy.

Run from the skill directory: python3 examples/tensors.py
Optional --output-root selects the parent of a fresh compare-tensors-UTC run.
The run contains contract.json, per-side manifests/arrays, validation reports
and report.json. Expected: 2 paired elements, L2 error 4, RMSE sqrt(8),
cosine 0.6 and maximum absolute error 4 at logical weight index 1.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_tensors import compare
from validate_artifacts import validate


def generate(output_root):
    root = Path(output_root) / datetime.now(timezone.utc).strftime('compare-tensors-%Y%m%dT%H%M%SZ')
    root.mkdir(parents=True, exist_ok=False)

    def write(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        return path

    common = dict(model='synthetic-weight', inputs='synthetic-batch',
                  numerics='float64', scope='gradients', state='synthetic-initial')
    contract = write(root / 'contract.json', dict(
        schema_version=2, kind='tensors', common=common, thresholds=None,
        expected_ranks={'a': [0], 'b': [0]}, stage='gradients',
        tensors=[dict(name='weight', shape=[2], present=True)]))
    manifests = []
    for side, values in [('a', [3., 4.]), ('b', [3., 0.])]:
        folder = root / side
        folder.mkdir()
        np.save(folder / 'gradient.npy', np.array(values, dtype=np.float64), allow_pickle=False)
        write(folder / 'environment.json', dict(rank=0, versions={'numpy': np.__version__}, actual=common))
        write(folder / 'status.json', dict(rank=0, status='completed'))
        manifest = write(folder / 'manifest.json', dict(
            schema_version=2, kind='tensors', common=common, side=side,
            expected_ranks=[0], ranks=[dict(rank=0, environment='environment.json', status='status.json')],
            stage='gradients', tensors=[dict(name='weight', shape=[2], present=True,
                                           shards=[dict(offset=0, value={'path': 'gradient.npy'})])]))
        checked = validate(contract, manifest)
        write(folder / 'validation.json', checked)
        if checked['status'] != 'validated':
            raise ValueError(checked)
        manifests.append(manifest)
    report = compare(contract, *manifests)
    write(root / 'report.json', report)
    if report['status'] != 'no_thresholds':
        raise ValueError(report)
    return root, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path(tempfile.gettempdir()))
    args = parser.parse_args()
    root, report = generate(args.output_root)
    print(json.dumps(dict(artifacts=str(root.resolve()), metrics=report['global']), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
