"""Run the pinned MindSpeed-LLM and Megatron GPT dataset paths on small CPU tensors."""

import ast
import subprocess
from functools import wraps
from types import SimpleNamespace

import numpy as np

PROJECT = "Ascend/MindSpeed-LLM"
BEFORE = "dd5ee0321c1179aaaded4fba4f119ed310b30639"
AFTER = "ee7ab917d786a3b073c0e64c24c1db2e0c2efce0"
MEGATRON = "a845aa7e12b3a117e24c2352b9e3e60bad2e3a17"
MIND_SPEED_DATASET = "mindspeed_llm/core/datasets/gpt_dataset.py"
MEGATRON_DATASET = "megatron/core/datasets/gpt_dataset.py"


def _git_module(checkout, revision, path):
    try:
        source = subprocess.check_output(
            ["git", "-C", str(checkout), "-c", "protocol.allow=never", "show", f"{revision}:{path}"],
            text=True,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Missing pinned source {revision}:{path} in {checkout}: {error.stderr.strip()}") from error
    return ast.parse(source, filename=path)


def _load_functions(module, names, namespace, class_name=None):
    scope = module
    if class_name is not None:
        scope = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    functions = [node for node in scope.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise ValueError(f"Missing upstream functions {names} in {class_name or 'module'}")
    code = ast.parse("from __future__ import annotations")
    code.body.extend(functions)
    exec(compile(code, "<pinned dataset source>", "exec"), namespace)


def _dataset_loss_mask(args):
    import torch

    checkout = args.source.resolve()
    mind_speed_revision = BEFORE if args.variant == "before" else AFTER
    mind_speed = _git_module(checkout, mind_speed_revision, MIND_SPEED_DATASET)
    megatron = _git_module(checkout / "Megatron-LM", MEGATRON, MEGATRON_DATASET)
    namespace = {
        "torch": torch,
        "numpy": np,
        "wraps": wraps,
        "get_args": lambda: SimpleNamespace(return_document_ids=False),
    }
    _load_functions(megatron, ("_get_ltor_masks_and_position_ids",), namespace)
    _load_functions(megatron, ("__getitem__",), namespace, class_name="GPTDataset")
    namespace["megatron_gpt_dataset"] = SimpleNamespace(
        _get_ltor_masks_and_position_ids=namespace["_get_ltor_masks_and_position_ids"],
    )
    function_names = ["gpt_dataset_getitem_wrapper"]
    if args.variant == "after":
        function_names.append("_getitem_with_cached_masks")
    _load_functions(mind_speed, function_names, namespace)
    getitem = namespace["gpt_dataset_getitem_wrapper"](namespace["__getitem__"])

    # Replace document storage and runtime configuration only. Both revisions
    # execute the actual wrapper, dataset path and mask builder on CPU.
    documents = ([10, 11, 12, -1, -1], [20, 21, 22, 23, 24])
    dataset = SimpleNamespace(
        config=SimpleNamespace(
            add_extra_token_to_sequence=True,
            tokenizer=SimpleNamespace(eod=99),
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            create_attention_mask=True,
        ),
        _pad_token_id=-1,
        masks_and_position_ids_are_cacheable=True,
        masks_and_position_ids_are_cached=False,
        _query_document_sample_shuffle_indices=lambda idx: (np.array(documents[idx]), np.array([idx])),
    )
    candidate = {}
    for key, idx in (("padded_loss_mask", 0), ("next_full_loss_mask", 1), ("batch_padding_loss_mask", None)):
        candidate[key] = getitem(dataset, idx)["loss_mask"].detach().cpu().numpy().copy()
    candidate["cached_loss_mask"] = dataset.cached_loss_mask.detach().cpu().numpy().copy()
    reference = {
        "padded_loss_mask": np.array([1, 1, 0, 0], dtype=np.float32),
        "next_full_loss_mask": np.ones(4, dtype=np.float32),
        "batch_padding_loss_mask": np.zeros(4, dtype=np.float32),
        "cached_loss_mask": np.ones(4, dtype=np.float32),
    }
    return reference, candidate


CASES = {
    "mindspeed-llm-gpt-dataset-loss-mask": {
        "title": "MindSpeed-LLM cached GPT loss mask isolation",
        "pr": "https://gitcode.com/Ascend/MindSpeed-LLM/merge_requests/5212",
        "project": PROJECT,
        "revision_before": BEFORE,
        "revision_after": AFTER,
        "dependencies": {"NVIDIA/Megatron-LM": MEGATRON},
        "upstream_status": "merged",
        "upstream_date": "2026-11-12",
        "modes": ("source-cpu",),
        "run": _dataset_loss_mask,
    },
}
