#!/usr/bin/env python3
"""Collect native CPU FP32/BF16 gradients and compare their corresponding arrays.

Run from the skill directory: python3 examples/compare_gradients.py
Requires Torch and NumPy. Both backward passes use the same classifier, initial
weights, features and labels; only precision differs. BF16 gradients are exported
losslessly as FP32, retaining native/export dtypes and effective inputs/state.
--output-root selects the parent of a fresh compare-gradients-UTC directory with
fp32.npz, bf16.npz, capture.json and report.json. Metrics have no tolerance.
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
from array_compare import compare_arrays


def generate(output_root):
    root = Path(output_root) / datetime.now(timezone.utc).strftime('compare-gradients-%Y%m%dT%H%M%SZ')
    root.mkdir(parents=True, exist_ok=False)
    features = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]]) / 8
    labels = torch.tensor([1, 2])
    initial = (torch.arange(12, dtype=torch.float32).reshape(3, 4) - 6) / 16
    common = dict(model='Linear(4,3,bias=False)',
                  inputs=dict(features=features.tolist(), labels=labels.tolist()),
                  numerics=dict(device='cpu', loss_dtype='torch.float32', reduction='mean'),
                  scope='gradients', state=dict(initial_weight=initial.tolist()))
    arrays, losses, actual = {}, {}, {}
    for side, dtype in dict(fp32=torch.float32, bf16=torch.bfloat16).items():
        model = torch.nn.Linear(4, 3, bias=False, dtype=dtype, device='cpu')
        with torch.no_grad():
            model.weight.copy_(initial)
        batch = features.to(dtype)
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction='mean')
        loss.backward()
        gradient = model.weight.grad
        arrays[side] = {'weight': gradient.detach().float().numpy()}
        np.savez(root / (side + '.npz'), **arrays[side])
        losses[side] = loss.item()
        actual[side] = dict(
            numerics=dict(device=str(logits.device), loss_dtype=str(loss.dtype), reduction='mean'),
            inputs=dict(features=batch.float().tolist(), labels=labels.tolist()),
            state=dict(initial_weight=model.weight.detach().float().tolist()),
            precision=dict(parameter=str(model.weight.dtype), forward=str(logits.dtype),
                           gradient=str(gradient.dtype), exported_gradient=str(arrays[side]['weight'].dtype)))
    expected = [name for name, _ in model.named_parameters()]
    report = compare_arrays(arrays['fp32'], arrays['bf16'], expected=expected, metrics=True)
    capture = dict(common=common, sides=actual, losses=losses, expected=expected,
                   versions=dict(torch=torch.__version__, numpy=np.__version__))
    for name, value in [('capture.json', capture), ('report.json', report)]:
        (root / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    if report['status'] != 'completed':
        raise ValueError(report)
    return root, losses, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path(tempfile.gettempdir()))
    args = parser.parse_args()
    root, losses, report = generate(args.output_root)
    print(json.dumps(dict(artifacts=str(root.resolve()), losses=losses,
                          coverage=report['coverage'], metrics=report['metrics']),
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
