#!/usr/bin/env python3
"""Compare native PyTorch gradients: FP32 versus BF16 on CPU.

Run from the skill directory: python3 examples/compare_gradients.py
Requires Torch and NumPy. A small classifier uses identical initial weights,
features and labels for two forward/backward passes. Only precision differs.
BF16 gradient values are exported losslessly as FP32; native dtypes are recorded.
--output-root selects the parent of a fresh compare-gradients-UTC run containing
captures, validation and report.json. Results are measured, with no thresholds.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_tensors import compare
from validate_artifacts import validate


def generate(output_root):
    root = Path(output_root) / datetime.now(timezone.utc).strftime('compare-gradients-%Y%m%dT%H%M%SZ')
    root.mkdir(parents=True, exist_ok=False)

    def write(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        return path

    features = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]]) / 8
    labels = torch.tensor([1, 2])
    initial = (torch.arange(12, dtype=torch.float32).reshape(3, 4) - 6) / 16
    common = dict(model='Linear(4,3,bias=False)',
                  inputs=dict(features=features.tolist(), labels=labels.tolist()),
                  numerics=dict(device='cpu', loss_dtype='torch.float32', reduction='mean'),
                  scope='gradients', state=dict(initial_weight=initial.tolist()))
    dtypes = dict(fp32=torch.float32, bf16=torch.bfloat16)
    sides = {side: dict(precision=dict(parameter=str(dtype), forward=str(dtype), gradient=str(dtype)))
             for side, dtype in dtypes.items()}
    contract = write(root / 'contract.json', dict(
        schema_version=2, kind='tensors', common=common, side_conditions=sides, thresholds=None,
        expected_ranks={side: [0] for side in sides}, stage='gradients',
        tensors=[dict(name='weight', shape=[3, 4], present=True)]))
    manifests, losses = [], {}
    for side, dtype in dtypes.items():
        model = torch.nn.Linear(4, 3, bias=False, dtype=dtype, device='cpu')
        with torch.no_grad():
            model.weight.copy_(initial)
        batch = features.to(dtype)
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction='mean')
        loss.backward()
        gradient = model.weight.grad
        losses[side] = loss.item()
        actual = {**common, 'numerics': dict(device=str(logits.device), loss_dtype=str(loss.dtype), reduction='mean'),
                  'inputs': dict(features=batch.float().tolist(), labels=labels.tolist()),
                  'state': dict(initial_weight=model.weight.detach().float().tolist()),
                  'precision': dict(parameter=str(model.weight.dtype), forward=str(logits.dtype),
                                    gradient=str(gradient.dtype))}
        folder = root / side
        folder.mkdir()
        np.save(folder / 'gradient.npy', gradient.detach().float().numpy(), allow_pickle=False)
        write(folder / 'environment.json', dict(rank=0, versions=dict(torch=torch.__version__, numpy=np.__version__), actual=actual))
        write(folder / 'status.json', dict(rank=0, status='completed', loss=losses[side]))
        manifest = write(folder / 'manifest.json', dict(
            schema_version=2, kind='tensors', common=common, side=side,
            expected_ranks=[0], ranks=[dict(rank=0, environment='environment.json', status='status.json')],
            stage='gradients', tensors=[dict(name='weight', shape=[3, 4], present=True,
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
    return root, losses, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path(tempfile.gettempdir()))
    args = parser.parse_args()
    root, losses, report = generate(args.output_root)
    print(json.dumps(dict(artifacts=str(root.resolve()), losses=losses,
                          metrics=report['global']), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
