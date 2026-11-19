"""Precision probes for three merged TorchNPU and op-plugin changes."""

import os

import numpy as np


def _execution_device(args, torch, backend=None):
    if args.mode == "cpu-smoke":
        return torch.device("cpu")
    if backend is not None:
        # Each native case has its own process. Set the backend before TorchNPU
        # initializes so default cannot inherit an unrelated caller setting.
        os.environ["TORCHINDUCTOR_NPU_BACKEND"] = backend
    import torch_npu  # noqa: F401  # Registers the NPU device and Inductor backend.

    device = torch.device(args.device)
    if device.type != "npu" or not torch.npu.is_available():
        raise RuntimeError("native-npu requires an available Ascend NPU")
    return device


def _capture(tensor, dtype):
    if tensor.dtype != dtype:
        raise RuntimeError(f"Expected {dtype}, got {tensor.dtype}")
    return tensor.detach().cpu().numpy().copy()


def _expm1(args):
    import torch

    device = _execution_device(args, torch, "default")
    values = np.array([-2**-26, -2**-30, 0, 2**-30, 2**-26], dtype=np.float32)
    x = torch.from_numpy(values).to(device)
    op = torch.expm1
    if args.mode == "native-npu":
        op = torch.compile(op, backend="inductor", fullgraph=True, options={"npu_backend": "default"})
    return {"expm1": np.expm1(values.astype(np.float64)).astype(np.float32)}, {
        "expm1": _capture(op(x), torch.float32),
    }


def _division(args):
    import torch

    device = _execution_device(args, torch, "triton_experimental")
    values = np.array([-(2**40) - 1, -5, -1, 0, 7, 8, 2**40 + 1], dtype=np.int64)
    x = torch.from_numpy(values).to(device)
    reference = {}
    candidate = {}
    compile_options = {"npu_backend": "triton_experimental"}
    for mode in ("floor", "trunc"):
        def divide(a, b, rounding_mode=mode):
            return torch.div(a, b, rounding_mode=rounding_mode)

        op = divide
        if args.mode == "native-npu":
            op = torch.compile(divide, backend="inductor", fullgraph=True, options=compile_options)
        for divisor in (3, -3):
            y = torch.tensor(divisor, dtype=torch.int64)
            if mode == "floor":
                expected = values // divisor
            else:
                expected = (np.abs(values) // abs(divisor)) * np.sign(values) * np.sign(divisor)
            for location, actual_y in (("host", y), ("device", y.to(device))):
                name = f"{mode}_{'pos' if divisor > 0 else 'neg'}_{location}"
                reference[name] = expected
                candidate[name] = _capture(op(x, actual_y), torch.int64)

    def fused(a):
        return torch.floor_divide(a + 1, 3) + 2

    op = fused
    if args.mode == "native-npu":
        op = torch.compile(fused, backend="inductor", fullgraph=True, options=compile_options)
    fused_values = np.array([-(2**40), 2**40], dtype=np.int64)
    reference["fused"] = (fused_values + 1) // 3 + 2
    candidate["fused"] = _capture(op(torch.from_numpy(fused_values).to(device)), torch.int64)
    return reference, candidate


def _mul(args):
    import torch

    device = _execution_device(args, torch)
    scalar = torch.tensor(0.224018394947052, dtype=torch.float32)
    reference = {}
    candidate = {}
    for name, dtype, value in (
        ("bf16", torch.bfloat16, -3.796875),
        ("fp16", torch.float16, 1.021484375),
    ):
        cpu_operand = torch.tensor([value], dtype=dtype)
        operand = cpu_operand.to(device)
        expected = (scalar.double() * cpu_operand.double()).to(dtype)
        expected_bits = _capture(expected.view(torch.int16), torch.int16).view(np.uint16)
        if args.mode == "cpu-smoke":
            # CPU mixed-dtype mul may pre-round the scalar. Use explicit CPU
            # FP32 arithmetic here to check the CUDA-style opmath reference.
            operand = operand.float()
        results = {
            "mul": torch.mul(scalar, operand),
            "mul_out": torch.mul(scalar, operand, out=torch.empty_like(operand)),
        }
        for entry, actual in results.items():
            if args.mode == "cpu-smoke":
                actual = actual.to(dtype)
            if actual.dtype != dtype:
                raise RuntimeError(f"Expected {dtype} from {entry}, got {actual.dtype}")
            key = f"{name}_{entry}_bits"
            reference[key] = expected_bits
            candidate[key] = _capture(actual.detach().cpu().view(torch.int16), torch.int16).view(np.uint16)
    return reference, candidate


CASES = {
    "torchnpu-inductor-expm1": {
        "title": "TorchNPU default Triton Inductor expm1 near zero",
        "pr": "https://gitcode.com/Ascend/pytorch/merge_requests/49365",
        "project": "Ascend/pytorch",
        "revision_before": "da5475419dacfba92868249b5b14bb1a4644be3e",
        "revision_after": "48b7e061fd3b12124f9f453fadfd9336383a2ee4",
        "upstream_status": "merged",
        "upstream_date": "2026-11-18",
        "run": _expm1,
        "modes": ("native-npu", "cpu-smoke"),
        "rtol": 1e-6,
    },
    "torchnpu-triton-int64-div": {
        "title": "TorchNPU experimental Triton int64 floor and trunc division",
        "pr": "https://gitcode.com/Ascend/pytorch/merge_requests/49540",
        "project": "Ascend/pytorch",
        "revision_before": "694d1b2fa62449c038108e05107801210a3f7f65",
        "revision_after": "fe7dfe2e3827fadb3478e73259f4599dbfb996e7",
        "upstream_status": "merged",
        "upstream_date": "2026-11-17",
        "run": _division,
        "modes": ("native-npu", "cpu-smoke"),
    },
    "op-plugin-mul-host-scalar": {
        "title": "op-plugin CPU float32 scalar times a BF16 or FP16 NPU tensor",
        "pr": "https://gitcode.com/Ascend/op-plugin/merge_requests/6062",
        "project": "Ascend/op-plugin",
        "revision_before": "1ba5f6842da2739d63833283ce2f56baad7fbf1a",
        "revision_after": "cb832887c0dfefff9109ee04b6a98765157a980a",
        "upstream_status": "merged",
        "upstream_date": "2026-10-27",
        "run": _mul,
        "modes": ("native-npu", "cpu-smoke"),
    },
}
