"""Capture precision witnesses from pinned vLLM-Ascend source or custom operators."""

import ast
import logging
import re
import subprocess
from importlib import import_module
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Iterable

import numpy as np

PROJECT = "vllm-project/vllm-ascend"
SINK_REVISIONS = {
    "before": "cb629ea1925a280d57b41b48416a85097f6cdb6b",
    "after": "db9ad13183f6836f7f83304590e1ba4035b888e5",
}
QSFA_REVISIONS = {
    "before": "fabfbfdac96199fe861895f29aceac11cb36587d",
    "after": "9b6770e239b1f919d451c63a290a7ef09e8c9dc1",
}


def _source(args) -> Path:
    if args.source is None:
        raise ValueError("--source must point to a vLLM-Ascend Git checkout")
    return args.source.resolve()


def _git(source: Path, *command: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), "-c", "protocol.allow=never", *command],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip())
    return result.stdout


def _sink_source(source: Path, revision: str, file_name: str) -> ast.Module:
    path = f"vllm_ascend/models/deepseek_v4/{file_name}.py"
    return ast.parse(_git(source, "show", f"{revision}:{path}"), filename=path)


def _source_functions(module, names, namespace, class_name=None):
    scope = module
    if class_name is not None:
        scope = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    functions = [node for node in scope.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise ValueError(f"Could not find source functions {names} in {class_name or 'module'}")
    # Postpone unrelated model annotations when loading functions without vLLM.
    tree = ast.parse("from __future__ import annotations")
    tree.body.extend(functions)
    exec(compile(tree, "<pinned vLLM-Ascend source>", "exec"), namespace)
    return namespace


def _attention_sink_reload(args):
    import torch

    source = _source(args)
    revision = SINK_REVISIONS[args.variant]
    model_source = _sink_source(source, revision, "model")

    @torch.no_grad()
    def copy_weight(param, value):
        param.copy_(value)

    namespace = {
        "torch": torch,
        "re": re,
        "Iterable": Iterable,
        "default_weight_loader": copy_weight,
        "fused_moe_make_expert_params_mapping": lambda *args, **kwargs: [],
        "get_ascend_config": lambda: SimpleNamespace(mix_placement=False),
        "get_tensor_model_parallel_rank": lambda: 1,
        "get_tensor_model_parallel_world_size": lambda: 2,
        "is_pp_missing_parameter": lambda *args: False,
        "logger": SimpleNamespace(info_once=logging.getLogger(__name__).info),
        "process_eagle_weight": lambda *args: None,
        "rocm_aiter_ops": SimpleNamespace(is_fusion_moe_shared_experts_enabled=lambda: False),
    }
    _source_functions(model_source, ("get_spec_layer_idx_from_weight_name",), namespace)
    initial = torch.arange(16, dtype=torch.float32)
    reference, candidate = {}, {}
    paths = (
        ("target", "model", "AscendDeepseekV4ForCausalLM", "model.layers.0.attn.attn_sink", 0, None),
        ("mtp", "mtp", "DeepSeekV4MTP", "mtp.0.attn.attn_sink", 0, "no_mtp_block_in_name"),
        ("dspark", "dspark", "DSparkDeepseekV4ForCausalLM", "mtp.1.attn.attn_sink", 4, "_remap_dspark_name"),
    )
    for route, file, class_name, checkpoint, layer_idx, name_helper in paths:
        names = ("load_weights",) if name_helper is None else ("load_weights", name_helper)
        route_source = model_source if file == "model" else _sink_source(source, revision, file)
        for dsa_cp in (False, True):
            context = dict(namespace, enable_dsa_cp=lambda: dsa_cp)
            _source_functions(route_source, names, context, class_name)
            model = torch.nn.Module()
            model.config = SimpleNamespace(num_attention_heads=16, num_hidden_layers=3, n_routed_experts=8)
            model.num_redundant_experts = 0
            model.quant_config = None
            model.rotation_path = None
            layer = torch.nn.Module()
            attention = torch.nn.Module()
            if route == "mtp":
                layer.mtp_block = torch.nn.Module()
                layer.mtp_block.self_attn = attention
            else:
                layer.self_attn = attention
            model.model = torch.nn.Module()
            model.model.layers = torch.nn.ModuleDict({str(layer_idx): layer})
            model.model.num_dspark_layers = 2
            model.model.get_expert_mapping = lambda: []
            if name_helper is not None:
                setattr(model, name_helper, MethodType(context[name_helper], model))

            indices = slice(None) if dsa_cp else slice(8, 16)
            storage = torch.nn.Parameter(torch.full_like(initial[indices], -1), requires_grad=False)
            attention.attn_sink = storage
            load_weights = MethodType(context["load_weights"], model)
            load_weights([(checkpoint, initial)])
            if not torch.equal(storage, initial[indices]):
                raise RuntimeError(f"Startup control failed for {route}, dsa_cp={dsa_cp}")
            for update, offset in enumerate((100, 200), start=1):
                payload = initial + offset
                # Model the reload protocol for this sink only. A meta copy does
                # not reach CPU storage. Weight-loader calls are buffered and replayed.
                pending = []
                deferred = torch.nn.Parameter(torch.empty_like(storage, device="meta"), requires_grad=False)

                def buffer_weight(param, value):
                    if param is not deferred:
                        raise RuntimeError("Reload called the sink loader with a different parameter")
                    pending.append(value.detach().clone())

                deferred.weight_loader = buffer_weight
                attention.attn_sink = deferred
                load_weights([(checkpoint, payload)])
                attention.attn_sink = storage
                for value in pending:
                    copy_weight(storage, value)
                scope = "dsa_cp" if dsa_cp else "tp2_rank1"
                key = f"{route}.{scope}.update{update}.attn_sink"
                reference[key] = payload[indices].numpy().copy()
                candidate[key] = storage.detach().numpy().copy()
    return reference, candidate


def _qsfa_query_tail(args):
    import torch
    import torch_npu

    source = _source(args)
    revision = QSFA_REVISIONS[args.variant]
    if _git(source, "rev-parse", "HEAD").strip() != revision:
        raise ValueError(f"--source must be checked out at {revision} for {args.variant}")
    utils = import_module("vllm_ascend.utils")
    package_path = source / "vllm_ascend"
    if Path(utils.__file__).resolve().parent != package_path:
        raise ValueError("Use an editable vLLM-Ascend installation of the selected checkout")
    torch_npu.npu.config.allow_internal_format = True
    device = args.device or "npu:0"
    torch.npu.set_device(device)
    device_config = import_module("vllm_ascend.device.device_config")
    hardware = import_module("vllm_ascend.device.hardware")
    utils.enable_custom_op()
    if device_config.get_ascend_device_type() == hardware.AscendDeviceType.A5:
        utils.bootstrap_custom_op_env()
        import_module("vllm_ascend.vllm_ascend_C")
    op = torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention_vllm

    actual, heads, dim, tile, topk = 36, 64, 512, 128, 1024
    block_num, block_size, nseq = 32, 128, 6

    def call(rows):
        # Preserve upstream's allocation order for the uninitialized-tail witness.
        generator = torch.Generator().manual_seed(1006)
        query = torch.randn((rows, heads, dim), generator=generator, dtype=torch.float32).to(torch.bfloat16)
        query[actual:] = 0
        query = query.to(device)
        values = torch.randint(-5, 10, (block_num, block_size, 1, dim), generator=generator, dtype=torch.int8)
        scales = torch.rand((block_num, block_size, 1, dim // tile), generator=generator, dtype=torch.float32)
        scales.uniform_(0.1, 1.0)
        kv = torch.cat((values, scales.contiguous().view(torch.int8)), dim=-1).contiguous().to(device)
        indices = torch.randint(0, block_num * block_size, (rows, 1, topk), generator=generator, dtype=torch.int32)
        indices[actual:] = 0
        indices = indices.to(device)
        block_table = torch.arange(block_num, dtype=torch.int32).reshape(1, block_num)
        block_table = block_table.expand(nseq, -1).contiguous().to(device)
        lengths_query = torch.tensor([6 * (i + 1) for i in range(nseq)], dtype=torch.int32).to(device)
        lengths_kv = torch.full((nseq,), block_num * block_size, dtype=torch.int32).to(device)
        output = op(
            query=query,
            key=kv,
            value=kv,
            sparse_indices=indices,
            scale_value=dim**-0.5,
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=lengths_query,
            actual_seq_lengths_kv=lengths_kv,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=tile,
            key_quant_mode=2,
            value_quant_mode=2,
            rope_head_dim=0,
            return_softmax_lse=False,
        )
        return output[0] if isinstance(output, tuple) else output

    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        # The upstream fixture uses the default CPU generator for scales.
        torch.random.default_generator.manual_seed(1006)
        call(128)
        output = call(64).cpu().float().numpy()
    if output.shape != (64, heads, dim):
        raise RuntimeError(f"Unexpected QSFA output shape: {output.shape}")
    tail = output[actual:]
    valid_finite = np.isfinite(output[:actual]).all(axis=(1, 2)).astype(np.float32)
    reference = {"padding_rows": np.zeros_like(tail), "valid_rows_are_finite": np.ones_like(valid_finite)}
    candidate = {"padding_rows": tail, "valid_rows_are_finite": valid_finite}
    return reference, candidate


CASES = {
    "vllm-ascend-deepseek-v4-sink-reload": {
        "title": "DeepSeek V4 attention sinks after layerwise reload",
        "pr": f"https://github.com/{PROJECT}/pull/18642",
        "project": PROJECT,
        "revision_before": SINK_REVISIONS["before"],
        "revision_after": SINK_REVISIONS["after"],
        "upstream_status": "merged",
        "upstream_date": "2026-11-03",
        "modes": ("source-cpu",),
        "run": _attention_sink_reload,
    },
    "vllm-ascend-qsfa-tnd-tail": {
        "title": "Quantized sparse flash attention TND padding after capacity shrink",
        "pr": f"https://github.com/{PROJECT}/pull/20578",
        "project": PROJECT,
        "revision_before": QSFA_REVISIONS["before"],
        "revision_after": QSFA_REVISIONS["after"],
        "upstream_status": "merged",
        "upstream_date": "2026-11-02",
        "modes": ("native-npu",),
        "run": _qsfa_query_tail,
    },
}
