"""Check the Mooncake sliding-window handoff using pinned scheduler and cache code."""

import ast
import logging
import subprocess
from types import MethodType, SimpleNamespace

import numpy as np

PROJECT = "vllm-project/vllm-ascend"
BEFORE = "a06547191910aa665c80eb914643d95552eeaafe"
AFTER = "47b32942d6e74b757fcae48cd40c181e53c7e588"
VLLM = "ba15685ab7f4f5fbed8630bbb7441f8837e76fbe"
SCHEDULER = "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_scheduler.py"
MANAGER = "vllm/v1/core/single_type_kv_cache_manager.py"
BLOCK_POOL = "vllm/v1/core/block_pool.py"
BLOCK_SIZE = 128
WINDOW = 512


def _read_module(checkout, revision, file):
    try:
        source = subprocess.check_output(
            ["git", "-C", str(checkout), "-c", "protocol.allow=never", "show", f"{revision}:{file}"],
            text=True,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Missing {revision}:{file} in {checkout}: {error.stderr.strip()}") from error
    return ast.parse(source, filename=file)


def _methods(module, class_name, names, bindings):
    scope = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    functions = [fn for fn in scope.body if isinstance(fn, ast.FunctionDef) and fn.name in names]
    if {fn.name for fn in functions} != set(names):
        raise ValueError(f"Could not find the pinned methods in {class_name}: {names}")
    code = ast.parse("from __future__ import annotations")
    code.body.extend(functions)
    namespace = dict(bindings)
    exec(compile(code, f"<pinned {class_name}>", "exec"), namespace)
    return {name: namespace[name] for name in names}


def _bind(target, methods):
    for name, method in methods.items():
        setattr(target, name, MethodType(method, target))


class _FreePages:
    def __init__(self, blocks):
        self.blocks = blocks

    @property
    def num_free_blocks(self):
        return len(self.blocks)

    def popleft_n(self, count):
        selected = self.blocks[:count]
        del self.blocks[:count]
        return selected

    def prepend_n(self, blocks):
        self.blocks[:0] = blocks

    def append_n(self, blocks):
        self.blocks.extend(blocks)


def _request(count, direction):
    return SimpleNamespace(
        prompt_token_ids=list(range(count)),
        _all_token_ids=list(range(count)),
        num_prompt_tokens=count,
        kv_transfer_params={direction: True},
        block_hashes=[],
        update_block_hashes=lambda: None,
    )


def _history(manager_methods, pool_methods, computed, query_position):
    null = SimpleNamespace(block_id=0, is_null=True)
    blocks = [
        SimpleNamespace(block_id=index, ref_cnt=0, is_null=False, pool=None, block_hash=None)
        for index in range(1, 16)
    ]
    pool = SimpleNamespace(
        free_block_queue=_FreePages(blocks),
        _reuse_watchers={},
        enable_caching=False,
        metrics_collector=None,
    )
    _bind(pool, pool_methods)
    manager = SimpleNamespace(
        block_size=BLOCK_SIZE,
        sliding_window=WINDOW,
        extra_retained_tokens=0,
        _partial_hit_reqs={},
        _record_new_block_ids=False,
        _null_block=null,
        req_to_blocks={"prompt": []},
        block_pool=pool,
    )
    _bind(manager, manager_methods)
    manager.allocate_new_blocks("prompt", computed, computed)
    table = manager.req_to_blocks["prompt"]
    # Fill pages with positional payload. Then overwrite every recycled page
    # so the retained-history check cannot accidentally accept stale content.
    payload = np.full((16, BLOCK_SIZE), -1, dtype=np.int64)
    for position in range(computed):
        page, offset = divmod(position, BLOCK_SIZE)
        payload[table[page].block_id, offset] = position
    manager.remove_skipped_blocks("prompt", computed)
    for block in pool.get_new_blocks(pool.get_num_free_blocks()):
        payload[block.block_id].fill(-1)
    start = max(0, query_position - WINDOW + 1)
    positions = np.arange(start, query_position, dtype=np.int64)
    observed = np.array(
        [payload[table[int(position) // BLOCK_SIZE].block_id, int(position) % BLOCK_SIZE] for position in positions],
        dtype=np.int64,
    )
    return positions, observed


def _swa_handoff(args):
    checkout = args.source.resolve()
    revision = BEFORE if args.variant == "before" else AFTER
    core = checkout / "vllm"
    scheduler_source = _read_module(checkout, revision, SCHEDULER)
    manager_source = _read_module(core, VLLM, MANAGER)
    pool_source = _read_module(core, VLLM, BLOCK_POOL)

    # Only cache-spec type tags, request state, payloads and the FIFO storage
    # are reduced here. The handoff, allocation and eviction are upstream code.
    sliding_spec = type("SlidingWindowSpec", (), {})
    scheduler_methods = _methods(
        scheduler_source,
        "MooncakeBaseConnectorScheduler",
        (
            "_needs_prefill_token_truncation",
            "_state_prefill_token_count",
            "_truncate_request_for_prefill",
            "on_new_request",
            "get_num_new_matched_tokens",
        ),
        {
            "SlidingWindowSpec": sliding_spec,
            "MambaSpec": type("MambaSpec", (), {}),
            "logger": logging.getLogger(__name__),
        },
    )
    manager_methods = _methods(
        manager_source,
        "SingleTypeKVCacheManager",
        ("allocate_new_blocks", "remove_skipped_blocks", "_remove_blocks_in_range"),
        {"cdiv": lambda numerator, denominator: -(-numerator // denominator)},
    )
    manager_methods.update(_methods(manager_source, "SlidingWindowManager", ("get_num_skipped_tokens",), {}))
    pool_methods = _methods(pool_source, "BlockPool", ("get_new_blocks", "get_num_free_blocks", "free_blocks"), {})

    reference, observed = {}, {}
    scenarios = (
        ("swa_1022", 1022, True),
        ("swa_1023", 1023, True),
        ("swa_1024", 1024, True),
        ("swa_single_token", 1, True),
        ("full_attention", 1023, False),
    )
    for label, prompt, is_sliding in scenarios:
        scheduler = SimpleNamespace(
            vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace())),
            group_unique_specs=[[sliding_spec() if is_sliding else object()]],
        )
        _bind(scheduler, scheduler_methods)
        scheduler.need_truncate = scheduler._needs_prefill_token_truncation()
        producer = _request(prompt, "do_remote_decode")
        consumer = _request(prompt, "do_remote_prefill")
        scheduler.on_new_request(producer)
        # The same request can be registered again without truncating it twice.
        scheduler.on_new_request(producer)
        scheduler.on_new_request(consumer)
        external_tokens, will_load = scheduler.get_num_new_matched_tokens(consumer, 0)
        completed = producer.num_prompt_tokens
        expected_prefill = prompt - 1 if is_sliding and prompt > 1 else prompt
        key = f"{label}.prefill_decode_external_load"
        reference[key] = np.array([expected_prefill, prompt, expected_prefill, 1], dtype=np.int64)
        observed[key] = np.array([completed, consumer.num_prompt_tokens, external_tokens, will_load], dtype=np.int64)
        if is_sliding and prompt > 1:
            key = f"{label}.kv_history"
            reference[key], observed[key] = _history(manager_methods, pool_methods, completed, prompt - 1)
    return reference, observed


CASES = {
    "vllm-ascend-mooncake-swa-handoff": {
        "title": "Mooncake sliding-window cache history at the prefill/decode handoff",
        "pr": f"https://github.com/{PROJECT}/pull/22270",
        "project": PROJECT,
        "revision_before": BEFORE,
        "revision_after": AFTER,
        "dependencies": {"vllm-project/vllm": VLLM},
        "upstream_status": "merged",
        "upstream_date": "2026-11-19",
        "modes": ("source-cpu",),
        "run": _swa_handoff,
    },
}
