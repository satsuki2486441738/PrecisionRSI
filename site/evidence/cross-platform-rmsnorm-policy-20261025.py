"""Case-local reproducible FP32 RMSNorm policy, with analytic RMSNorm backward.

Both peers compute their own statistics. Fixed pairwise additions replace native
reductions, and four explicitly ordered Newton steps replace native rsqrt. Model
formula, epsilon, weights and boundary dtypes stay unchanged. This changes numeric
policy; it is not a demonstrated upstream correctness bug fix.
"""
import types
import torch


def pairwise_sum(x):
    """Sum the last dimension with a fixed addition tree, padding odd widths."""
    while x.shape[-1] > 1:
        if x.shape[-1] % 2:
            x = torch.cat((x, torch.zeros_like(x[..., :1])), dim=-1)
        x = x[..., 0::2] + x[..., 1::2]
    return x


def inverse_sqrt(a):
    """Deterministic FP32 seed and refinement; requires positive normal FP32 a."""
    bits = a.detach().contiguous().view(torch.int32)
    half_bits = torch.div(bits, torch.full_like(bits, 2), rounding_mode='floor')
    r = (0x5F375A86 - half_bits).view(torch.float32)
    half = a * 0.5
    for _ in range(4):
        correction = half * r
        correction = correction * r
        r = r * (1.5 - correction)
    return r


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        xf = x.float()
        wf = weight.float()
        mean = pairwise_sum(xf * xf) / x.shape[-1]
        inv = inverse_sqrt(mean + eps)
        normalized = xf * inv
        y = normalized * (1.0 + wf)
        ctx.save_for_backward(xf, wf, inv)
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, w, inv = ctx.saved_tensors
        go = grad_output.float()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            v = go * (1.0 + w)
            dot = pairwise_sum(v * x) / x.shape[-1]
            projection = (x * inv) * inv
            dx = ((v - projection * dot) * inv).to(ctx.input_dtype)
        if ctx.needs_input_grad[1]:
            products = go * (x * inv)
            dw = pairwise_sum(products.reshape(-1, w.numel()).transpose(0, 1))
            dw = dw.reshape_as(w).to(ctx.weight_dtype)
        return dx, dw, None


class _BackwardOnlyRMSNorm(_RMSNorm):
    @staticmethod
    def forward(ctx, x, weight, eps):
        xf = x.float()
        wf = weight.float()
        inv = inverse_sqrt(pairwise_sum(xf * xf) / x.shape[-1] + eps)
        ctx.save_for_backward(xf, wf, inv)
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        # Exact source order of the unchanged native forward.
        native_inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return ((xf * native_inv) * (1.0 + wf)).to(x.dtype)


def rms_norm(x, weight, eps=1e-6, backward_only=False):
    """RMSNorm over the last axis; BF16/FP16/FP32 boundaries, FP32 internals."""
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f'Unsupported RMSNorm input dtype: {x.dtype}')
    if weight.ndim != 1 or x.shape[-1] != weight.numel() or eps <= 0:
        raise ValueError('Expected a matching one-dimensional weight and positive epsilon')
    function = _BackwardOnlyRMSNorm if backward_only else _RMSNorm
    return function.apply(x, weight, float(eps))


def install(model, scope='all', backward_only=False):
    record = dict(classification='numerical-policy candidate', scope=scope,
                  forward='unchanged native source order' if backward_only else 'fixed FP32 policy',
                  statistics='independent per peer', accumulation='fixed FP32 pairwise tree',
                  rsqrt='fixed integer seed and four FP32 Newton steps',
                  backward='analytic RMSNorm with fixed FP32 reductions', modules=[], calls=0)
    for name, module in model.named_modules():
        if type(module).__name__ != 'Qwen3_5RMSNorm':
            continue
        if scope == 'layer8' and not name.endswith('.layers.8.input_layernorm'):
            continue
        if scope not in ('all', 'layer8'):
            raise ValueError(scope)
        record['modules'].append(dict(name=name, width=module.weight.numel(), eps=module.eps,
                                      weight_dtype=str(module.weight.dtype),
                                      weight_requires_grad=module.weight.requires_grad))
        def forward(self, x):
            record['calls'] += 1
            return rms_norm(x, self.weight, self.eps, backward_only=backward_only)
        module.forward = types.MethodType(forward, module)
    if not record['modules']:
        raise ValueError('No matching Qwen3_5RMSNorm modules')
    return record
