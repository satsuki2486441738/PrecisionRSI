"""Reproduce MindSpeed-LLM indexer loss mutating saved sparse-attention indices."""

import ast
import subprocess
from types import SimpleNamespace

import numpy as np

DSA_INDEXER = "mindspeed_llm/tasks/models/transformer/dsa_indexer.py"
REVISIONS = {
    "before": "4458c30c2deff2534d5373c6002b8a16c8fb0551",
    "after": "259fa5991088c869d78c2e0f601a0fa096648c47",
}


def _load_indexer_loss(checkout, revision, torch):
    result = subprocess.run(
        ["git", "-C", str(checkout), "-c", "protocol.allow=never", "show", f"{revision}:{DSA_INDEXER}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"Fetch MindSpeed-LLM revision {revision} into {checkout} before running.")
    definitions = {
        node.name: node for node in ast.parse(result.stdout, filename=DSA_INDEXER).body
        if isinstance(node, ast.FunctionDef)
    }
    names = ("compute_dsa_indexer_loss_dsv4", "compute_indexer_loss")
    code = ast.Module(body=[definitions[name] for name in names], type_ignores=[])
    namespace = {
        "torch": torch,
        "F": torch.nn.functional,
        "get_args": lambda: SimpleNamespace(use_fused_lightning_indexer=False),
    }
    exec(compile(code, DSA_INDEXER, "exec"), namespace)
    return namespace[names[0]]


def _dsa_topk_padding(args):
    import torch

    indexer_loss = _load_indexer_loss(args.source, REVISIONS[args.variant], torch)
    values = np.array([1, 3, 5, 7], dtype=np.float32)
    query_weights = np.array([1, 2, 4], dtype=np.float32)
    reference, observed = {}, {}
    for label, rows in (
        ("padding", [[0, -1, -1], [1, 2, -1], [2, 3, 0]]),
        ("full_topk", [[0, 1, 3], [1, 2, 0], [2, 3, 0]]),
    ):
        topk = torch.tensor([rows], dtype=torch.int64)
        # The native caller can retain a BSND view of the original indices.
        saved = topk.view(1, 3, 1, 3)[0, :, 0, :]
        attention = torch.full((1, 3, 4), 0.25, dtype=torch.float32)
        index_logits = torch.zeros((1, 3, 3), dtype=torch.float32)
        loss = indexer_loss(attention, index_logits, topk, 1.0)

        # A small sparse gather consumer exposes the numerical effect of
        # changing the -1 contract. It is a CPU proxy for saved-view consumers.
        kv = torch.tensor(values, requires_grad=True)
        valid = saved != -1
        selected = kv[saved.clamp_min(0)].masked_fill(~valid, 0)
        weighted = selected.sum(dim=-1) * torch.from_numpy(query_weights)
        gradient, = torch.autograd.grad(weighted.sum(), kv)

        expected_output = np.zeros(3, dtype=np.float32)
        expected_gradient = np.zeros(4, dtype=np.float32)
        for query, row in enumerate(rows):
            for index in row:
                if index >= 0:
                    expected_output[query] += values[index] * query_weights[query]
                    expected_gradient[index] += query_weights[query]

        reference[f"{label}.saved_topk_indices"] = np.array(rows, dtype=np.int64)
        reference[f"{label}.proxy_output"] = expected_output
        reference[f"{label}.proxy_kv_grad"] = expected_gradient
        # Uniform target and indexer distributions have zero auxiliary loss.
        reference[f"{label}.uniform_indexer_loss"] = np.array(0, dtype=np.float32)
        observed[f"{label}.saved_topk_indices"] = saved.detach().numpy().copy()
        observed[f"{label}.proxy_output"] = weighted.detach().numpy()
        observed[f"{label}.proxy_kv_grad"] = gradient.detach().numpy()
        observed[f"{label}.uniform_indexer_loss"] = loss.detach().numpy()
    return reference, observed


CASES = {
    "mindspeed-llm-dsa-topk-padding": {
        "title": "MindSpeed-LLM DSA training loss mutates saved sparse-attention top-k indices",
        "pr": "https://gitcode.com/Ascend/MindSpeed-LLM/merge_requests/5235",
        "project": "Ascend/MindSpeed-LLM",
        "revision_before": REVISIONS["before"],
        "revision_after": REVISIONS["after"],
        "upstream_status": "open",
        "upstream_date": "2026-11-19",
        "modes": ("source-cpu",),
        "run": _dsa_topk_padding,
    },
}
