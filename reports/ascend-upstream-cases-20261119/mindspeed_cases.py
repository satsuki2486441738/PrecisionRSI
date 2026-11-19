"""Run MindSpeed Core training regressions from pinned Git source."""

import ast
import subprocess
from types import SimpleNamespace

import numpy as np

LINEAR_PATH = "mindspeed/te/pytorch/module/linear.py"
LINEAR_REVISIONS = {
    "before": "c2357e9ae4689cf1791a5f258217cd256ce41ce1",
    "after": "17f1bb727be109b152e5c2e4d74664592233b2df",
}

MTP_BEFORE = "43e884e651b6460b0a49a1eac5991d63eb4996a7"
MTP_AFTER = "5f4402805352459264fb526ca5f587453b811c72"
MTP_PATH = "mindspeed/core/transformer/multi_token_prediction.py"


def _read_source(checkout, revision, path):
    result = subprocess.run(
        [
            "git", "-C", str(checkout), "-c", "protocol.allow=never",
            "show", f"{revision}:{path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"Fetch MindSpeed revision {revision} into {checkout} before running.")
    return result.stdout


def _linear_source(checkout, revision):
    source = _read_source(checkout, revision, LINEAR_PATH)
    module = ast.parse(source, filename=LINEAR_PATH)
    definitions = {
        node.name: node for node in module.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    wrapper = next(
        node for node in definitions["TEColumnParallelLinear"].body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    # The wrapper and autograd run as written. Skip imports that initialize NPU
    # and Megatron, plus the module constructor that allocates distributed state.
    code = ast.parse("from __future__ import annotations")
    code.body.extend(definitions[name] for name in (
        "BackwardStateStorage", "ColumnParallelSeq", "reshape_to_2D",
        "_calculate_grad_weight", "_calculate_grad_bias", "calculate_grad",
    ))
    code.body.append(wrapper)
    return compile(code, LINEAR_PATH, "exec")


def _sp_column_bias(args):
    import torch

    code = _linear_source(args.source, LINEAR_REVISIONS[args.variant])
    # Binary fractions make the tiny BF16 fixture exact. Different rows and
    # rank-local weights make sequence ordering and reduction observable.
    inputs = torch.tensor([
        [1, 2, 3, 4], [2, 0, 1, 3], [0, 3, 1, 2], [4, 1, 0, 2],
    ], dtype=torch.bfloat16) / 4
    weights = torch.tensor([
        [[1, 2, 0, 1], [0, -2, 1, 1]], [[0, 1, 1, 2], [2, 1, -1, 0]],
    ], dtype=torch.bfloat16) / 4
    output_grads = torch.tensor([
        [[1, 0], [1, 2], [0, 2], [3, 1]], [[2, 3], [1, 0], [3, 1], [1, 2]],
    ], dtype=torch.bfloat16) / 4
    biases = torch.tensor([[2, -1], [1, -3]], dtype=torch.bfloat16)

    full_x = inputs.float().numpy()
    full_w = weights.float().numpy()
    full_dy = output_grads.float().numpy()
    dx_shards = np.split(np.einsum("rtj,rji->ti", full_dy, full_w), 2)
    x_shards = inputs.chunk(2)
    handle = SimpleNamespace(wait=lambda: None)
    reference, candidate = {}, {}

    for rank in range(2):
        peer = 1 - rank

        def allgather_matmul(local_x, local_w, matmul_bias, fp8_meta, key, fp8_enabled):
            if matmul_bias is not None or fp8_enabled:
                raise ValueError("The CPU fixture expects bias-free BF16 all-gather matmul.")
            slices = list(x_shards)
            slices[rank] = local_x
            gathered = torch.cat(slices, dim=0)
            return gathered @ local_w.t(), gathered, None

        def reduce_scatter(output, partial, *, group, async_op):
            peer_partial = output_grads[peer] @ weights[peer]
            output.copy_((partial + peer_partial).chunk(2)[rank])
            return handle

        namespace = {
            "torch": SimpleNamespace(
                autograd=torch.autograd, empty=torch.empty,
                distributed=SimpleNamespace(_reduce_scatter_base=reduce_scatter),
            ),
            "get_args": lambda: SimpleNamespace(gradient_accumulation_fusion=False),
            "get_ops": lambda: SimpleNamespace(allgather_matmul=allgather_matmul),
            "get_tensor_model_parallel_group": lambda: None,
            "COMM_OVERLAP_CONFIG": SimpleNamespace(save_allgather_input=True),
            "MatmulKey": SimpleNamespace(forward="forward"),
            "DummyHandle": handle,
            "is_fp8_tensor_2d": lambda tensor: False,
        }
        exec(code, namespace)
        for bias_kind, skip in (("absent", False), ("zero", False),
                                ("inline", False), ("deferred", True)):
            bias = None
            ref_bias = None
            if bias_kind != "absent":
                initial = biases[rank]
                if bias_kind == "zero":
                    initial = torch.zeros_like(initial)
                ref_bias = initial.float().numpy()
                bias = initial.clone().requires_grad_()
            x = x_shards[rank].clone().requires_grad_()
            weight = weights[rank].clone().requires_grad_()
            layer = SimpleNamespace(
                weight=weight, bias=bias, skip_bias_add=skip, explicit_expert_comm=False,
                sequence_parallel=True, fp8_meta=SimpleNamespace(is_fp8_enable=lambda: False),
            )
            output, deferred_bias = namespace["forward"](layer, x)
            if (deferred_bias is not None) != skip:
                raise ValueError("Unexpected presence of a deferred bias.")

            rank_w, rank_dy = full_w[rank], full_dy[rank]
            unbiased = full_x @ rank_w.T
            biased = unbiased if ref_bias is None else unbiased + ref_bias
            effective_output = output if deferred_bias is None else output + deferred_bias
            effective_output.backward(output_grads[rank])

            key = f"rank{rank}.{bias_kind}"
            expected = {
                "output": unbiased if skip else biased,
                "input_grad": dx_shards[rank],
                "weight_grad": rank_dy.T @ full_x,
            }
            actual = {"output": output, "input_grad": x.grad, "weight_grad": weight.grad}
            if bias is not None:
                expected["bias_grad"] = rank_dy.sum(axis=0)
                actual["bias_grad"] = bias.grad
            if skip:
                expected["caller_output"] = biased
                expected["deferred_bias"] = ref_bias
                actual["caller_output"] = effective_output
                actual["deferred_bias"] = deferred_bias
            reference.update({f"{key}.{name}": array for name, array in expected.items()})
            candidate.update({f"{key}.{name}": value.detach().float().numpy()
                              for name, value in actual.items()})
    return reference, candidate


def _load_mtp_roll(checkout, revision):
    source = _read_source(checkout, revision, MTP_PATH)
    namespace = {}
    exec(compile(source, MTP_PATH, "exec", dont_inherit=True), namespace)

    def unexpected_fallback(*args, **kwargs):
        raise RuntimeError("Expected the MindSpeed CP=1 packed roll path")

    return namespace["roll_tensor_packed_seq_wrapper"](unexpected_fallback)


def _training_observables(torch, labels, loss_mask, token_count):
    logits = (torch.arange(labels.numel() * 7, dtype=torch.float32).reshape(-1, 7) / 8).requires_grad_()
    token_losses = torch.nn.functional.cross_entropy(logits, labels.flatten(), reduction="none")
    mean_loss = (token_losses * loss_mask.flatten()).sum() / token_count
    logits_grad, = torch.autograd.grad(mean_loss, logits)
    return {
        "labels": labels,
        "loss_mask": loss_mask,
        "valid_tokens": token_count,
        "mean_loss": mean_loss,
        "logits_grad": logits_grad,
    }


def _packed_mtp(args):
    import torch

    roll = _load_mtp_roll(args.source, MTP_BEFORE if args.variant == "before" else MTP_AFTER)
    reference, observed = {}, {}
    for name, boundaries, shifted_labels, shifted_mask in (
        ("two_segments", [3, 6], [2, 3, 0, 5, 6, 0], [1, 1, 0, 1, 1, 0]),
        ("one_segment", [6], [2, 3, 4, 5, 6, 0], [1, 1, 1, 1, 1, 0]),
    ):
        endpoints = torch.tensor(boundaries, dtype=torch.int64)
        labels = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int64)
        mask = torch.ones(1, 6)
        packed = SimpleNamespace(cu_seqlens_q=endpoints, cu_seqlens_kv=endpoints, _mindspeed_endpoints=True)
        # Mirror the adaptor's endpoint marker, while exercising the unmodified
        # MTP wrapper with the same packed inputs at both revisions.
        actual_labels, _ = roll(labels, -1, -1, packed)
        actual_mask, actual_count = roll(mask, -1, -1, packed)
        if packed.cu_seqlens_q is not endpoints or packed.cu_seqlens_kv is not endpoints:
            raise RuntimeError("The packed roll replaced attention's shared endpoints")

        expected = _training_observables(
            torch,
            torch.tensor([shifted_labels], dtype=torch.int64),
            torch.tensor([shifted_mask], dtype=torch.float32),
            torch.tensor(sum(shifted_mask), dtype=torch.float32),
        )
        expected["attention_endpoints"] = torch.tensor(boundaries, dtype=torch.int64)
        actual = _training_observables(torch, actual_labels, actual_mask, actual_count)
        actual["attention_endpoints"] = endpoints
        for key, value in expected.items():
            reference[f"{name}_{key}"] = value.detach().numpy().copy()
            observed[f"{name}_{key}"] = actual[key].detach().numpy().copy()
    return reference, observed



CASES = {
    "mindspeed-te-sp-column-bias": {
        "title": "MindSpeed TE sequence-parallel column linear drops a nonzero bias",
        "pr": "https://gitcode.com/Ascend/MindSpeed/merge_requests/4128",
        "project": "Ascend/MindSpeed",
        "revision_before": LINEAR_REVISIONS["before"],
        "revision_after": LINEAR_REVISIONS["after"],
        "upstream_status": "open",
        "upstream_date": "2026-11-02",
        "modes": ("source-cpu",),
        "run": _sp_column_bias,
    },
    "mindspeed-packed-mtp-first-segment": {
        "title": "MindSpeed CP1 packed MTP shifts the first segment and masks its final token",
        "pr": "https://gitcode.com/Ascend/MindSpeed/merge_requests/4115",
        "project": "Ascend/MindSpeed",
        "revision_before": MTP_BEFORE,
        "revision_after": MTP_AFTER,
        "upstream_status": "merged",
        "upstream_date": "2026-10-30",
        "run": _packed_mtp,
        "modes": ("source-cpu",),
    },
}
