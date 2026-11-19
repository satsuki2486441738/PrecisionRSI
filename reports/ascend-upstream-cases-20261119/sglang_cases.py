"""Run the affected SGLang source against small independent CPU references."""

import __future__
import ast
import subprocess
from types import SimpleNamespace

import numpy as np

MLA_BEFORE = "2120cf1190d32135358ea0fa73a87c4ca232803f"
MLA_AFTER = "643909b5a898665aa7640c85b48963598f41815a"
GDN_BEFORE = "c2578f28cfaece96167765ed93cab9a484f6fe95"
GDN_AFTER = "8e5404fa8b8a1e12f7029c7e7c17dbfed7f89475"
NPU_ATTENTION = "python/sglang/srt/hardware_backend/npu/attention"
GDN_TRITON = "python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py"


def _read_source(checkout, revision, path):
    if checkout is None:
        raise ValueError(
            "Supply --source with a local SGLang Git checkout containing both revisions."
        )
    result = subprocess.run(
        [
            "git", "-C", str(checkout), "-c", "protocol.allow=never",
            "show", f"{revision}:{path}",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        raise ValueError(
            f"Cannot read {revision}:{path} from {checkout}. "
            "Fetch the pinned revision before running."
        )
    return result.stdout


def _load_method(checkout, revision, path, class_name, method_name, namespace):
    # Importing the full module initializes accelerator backends. Compile the
    # unmodified method body with explicit bindings only for dependencies used.
    module = ast.parse(_read_source(checkout, revision, path), filename=path)
    owner = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    code = compile(
        ast.Module(body=[method], type_ignores=[]),
        path,
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, namespace)
    return namespace[method_name]


def _mla_prefix(args):
    import torch

    revision = MLA_BEFORE if args.variant == "before" else MLA_AFTER
    cache_path = f"{NPU_ATTENTION}/mla_cache.py"
    functions = {}
    exec(
        compile(
            _read_source(args.source, revision, cache_path),
            cache_path,
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        functions,
    )
    reference, candidate = {}, {}
    for layout in ("nd", "nz"):
        is_nz = layout == "nz"
        load_prefix = _load_method(
            args.source,
            revision,
            f"{NPU_ATTENTION}/ascend_backend.py",
            "AscendAttnBackend",
            "_load_mla_prefix_cache",
            {**functions, "is_fia_nz": lambda: is_nz},
        )
        for fixture, lengths in (("mixed", (3, 0, 5, 1)), ("aligned", (4, 0, 8, 4))):
            page_size, head_dim = 4, 32
            blocks = [3, 1, 3, 0]
            latent = torch.arange(4 * page_size * head_dim, dtype=torch.float32)
            latent = latent.reshape(4, page_size, 1, head_dim)
            rope = -latent - 1
            buffers = (latent, rope)
            if is_nz:
                buffers = tuple(
                    pages.reshape(4, page_size, head_dim // 16, 16)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .view_as(pages)
                    for pages in buffers
                )
            metadata = SimpleNamespace(
                flatten_prefix_block_tables=torch.tensor(blocks, dtype=torch.int32),
                prefix_token_indices=None,
            )
            if revision == MLA_AFTER and any(length % page_size for length in lengths):
                build_token_indices = functions["build_mla_prefix_token_indices"]
                metadata.prefix_token_indices = build_token_indices(
                    lengths, page_size, device=torch.device("cpu"),
                )
            backend = SimpleNamespace(
                token_to_kv_pool=SimpleNamespace(
                    get_key_buffer=lambda layer_id: buffers[0],
                    get_value_buffer=lambda layer_id: buffers[1],
                ),
                forward_metadata=metadata,
                dcp_size=1,
            )
            actual_latent, actual_rope = load_prefix(backend, SimpleNamespace(layer_id=0))
            for name, pages, out in (("latent", latent, actual_latent), ("rope", rope, actual_rope)):
                # Index valid logical tokens independently of the physical layout.
                rows = []
                offset = 0
                for length in lengths:
                    for index in range(length):
                        page = blocks[offset + index // page_size]
                        rows.append(pages[page, index % page_size])
                    offset += (length + page_size - 1) // page_size
                expected = torch.stack(rows)
                flattened = out.reshape(-1, 1, head_dim)
                key = f"{fixture}.{layout}.{name}"
                reference[key] = expected.numpy()
                # FIA consumes the first sum(lengths) rows using the logical boundaries.
                candidate[key] = flattened[: expected.shape[0]].numpy()
                reference[f"{key}.rows"] = np.array([expected.shape[0]])
                candidate[f"{key}.rows"] = np.array([flattened.shape[0]])
    return reference, candidate


def _chunk_kernel(
    *,
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    cu_seqlens,
    head_first,
    use_qk_l2norm_in_kernel,
):
    """CPU stand-in for the external chunk kernel's [K,V] layout contract."""
    import torch

    assert not head_first and not use_qk_l2norm_in_kernel
    assert initial_state.is_contiguous()
    assert tuple(initial_state.shape) == (
        cu_seqlens.numel() - 1, v.shape[-2], k.shape[-1], v.shape[-1]
    )
    output, final, chunks = [], [], []
    boundaries = cu_seqlens.tolist()
    scale = k.shape[-1] ** -0.5
    for sequence, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
        state = initial_state[sequence].clone()
        for time in range(start, stop):
            if (time - start) % 64 == 0:
                chunks.append(state.clone())
            state = state * g[0, time].exp()[:, None, None]
            predicted = torch.einsum("hkv,hk->hv", state, k[0, time])
            residual = (v[0, time] - predicted) * beta[0, time, :, None]
            state = state + torch.einsum("hk,hv->hkv", k[0, time], residual)
            output.append(torch.einsum("hkv,hk->hv", state, q[0, time]) * scale)
        final.append(state)
    return (
        torch.stack(output).unsqueeze(0),
        torch.stack(final),
        torch.stack(chunks).unsqueeze(0),
    )


def _gdn_layout(args):
    import torch

    revision = GDN_BEFORE if args.variant == "before" else GDN_AFTER
    extend = _load_method(
        args.source,
        revision,
        GDN_TRITON,
        "TritonGDNKernel",
        "extend",
        {
            "is_cpu": lambda: False,
            "is_npu": lambda: True,
            "is_xpu": lambda: False,
            "chunk_gated_delta_rule": _chunk_kernel,
        },
    )
    lengths = (65, 3)
    boundaries = (0, lengths[0], sum(lengths))
    slots = (2, 0)
    heads, head_dim = 2, 4
    times = torch.arange(sum(lengths))
    k = torch.nn.functional.one_hot(times % head_dim, head_dim).float()
    q = torch.nn.functional.one_hot((times + 1) % head_dim, head_dim).float()
    q = q[None, :, None, :].repeat(1, 1, heads, 1)
    k = k[None, :, None, :].repeat(1, 1, heads, 1)
    v = (torch.arange(sum(lengths) * heads * head_dim) % 19 - 9).float() / 8
    v = v.reshape(1, sum(lengths), heads, head_dim)
    state = torch.arange(3 * heads * head_dim**2, dtype=torch.float32) / 8
    state = state.reshape(3, heads, head_dim, head_dim)
    g = torch.zeros((1, sum(lengths), heads))
    beta = torch.full_like(g, 0.5)
    actual = extend(
        None,
        q,
        k,
        v,
        g,
        beta,
        ssm_states=state,
        cache_indices=torch.tensor(slots),
        query_start_loc=torch.tensor(boundaries),
        use_qk_l2norm_in_kernel=False,
    )

    # With one-hot q/k, g=0 and beta=0.5, a direct [V,K] reference updates
    # one state column per step, then reads one query column scaled by 1/2.
    output, final, chunks = [], [], []
    values = v.numpy()
    for slot, start, stop in zip(slots, boundaries, boundaries[1:]):
        state_vk = state[slot].numpy().copy()
        for time in range(start, stop):
            if (time - start) % 64 == 0:
                chunks.append(state_vk.copy())
            column = time % head_dim
            state_vk[:, :, column] += (values[0, time] - state_vk[:, :, column]) * 0.5
            output.append(state_vk[:, :, (time + 1) % head_dim] * 0.5)
        final.append(state_vk)
    names = ("attention", "final_state_vk", "chunk_state_vk")
    expected = (np.stack(output)[None], np.stack(final), np.stack(chunks)[None])
    return dict(zip(names, expected)), dict(zip(names, [tensor.numpy() for tensor in actual]))


CASES = {
    "sglang-mla-prefix-compaction": {
        "title": "SGLang Ascend MLA cache includes invalid page tails in mixed batches",
        "pr": "https://github.com/sgl-project/sglang/pull/47924",
        "project": "sgl-project/sglang",
        "revision_before": MLA_BEFORE,
        "revision_after": MLA_AFTER,
        "upstream_status": "merged",
        "upstream_date": "2026-11-04",
        "run": _mla_prefix,
        "modes": ("source-cpu",),
    },
    "sglang-gdn-state-layout": {
        "title": "SGLang Ascend GDN prefill passes recurrent state in the wrong layout",
        "pr": "https://github.com/sgl-project/sglang/pull/50245",
        "project": "sgl-project/sglang",
        "revision_before": GDN_BEFORE,
        "revision_after": GDN_AFTER,
        "upstream_status": "open",
        "upstream_date": "2026-11-13",
        "run": _gdn_layout,
        "modes": ("source-cpu",),
    },
}
