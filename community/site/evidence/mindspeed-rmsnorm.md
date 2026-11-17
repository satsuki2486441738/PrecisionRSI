# Historical MindSpeed RMSNorm repair

Date: 2026-10-21. Model: Qwen3.5-9B-Base. Hardware: Ascend 910C, TP4.

## Cause and repair

MindSpeed commit `ff884722a3ce7079fbafe3cb97a3287311f58a5b`, used with Megatron Bridge 0.4.2, ignores `layernorm_zero_centered_gamma=True`. The effective scale must be `1 + stored gamma`. The repair honors this convention during initialization and in both forward paths while preserving checkpoint representation and native kernels.

Local repair commit: `9ca343aaa1a2ca0688560b2f3cf8b0b8be38be38`. Not submitted upstream.

## Verification

- Original source matches pinned upstream bytes; original scores reproduce historical NPU scores exactly.
- Original CPU regression: four methods, seven failing assertions/subtests. Fixed CPU regression: four methods pass.
- Actual NPU forward/backward and checkpoint checks pass; the CPU interception test is skipped on NPU.
- 2,608 initial parameter records match in name, shape, dtype and SHA256.
- Each rank records 46 native fused RMSNorm calls. Coverage: 32/32 actions.
- Before and after use the same retained CUDA observation.

| Metric (nat/token) | Before | After |
| --- | ---: | ---: |
| mae_nat | 10.637649923563004 | 0.026155292987823486 |
| p99_nat | 13.878965330123902 | 0.10877090454101565 |
| max_abs_nat | 13.952625274658203 | 0.11327171325683594 |
| nll_abs_diff_nat_per_token | 10.637649923563004 | 0.00041621923446655273 |

P99 uses global linear interpolation: `h = 0.99 * (N - 1)`.

Whole-model verification covers forward/scoring and the derived objective. Backward and checkpoint tests are operator-level. Residual differences remain unattributed. Numerical acceptance is pending user judgment.

## Evidence identities

- Model revision: `68c46c4b3498877f3ef123c856ecfde50c39f404`.
- Original Norm SHA256: `37aa465dddf87488216e5a32fd3247622be0e344aec814f825c4b4142c9264da`.
- Corrected Norm SHA256: `b886c2134f89d9a8c89161121b84bc82d429242c51732612a7d90b45b2fa545e`.
- Patch SHA256: `fa73c377bb44ca17bdb2b5090666fce94a046de840072f0122aa702c6ff2a4de`.
- Private evidence bundle SHA256: `04e573df82347652f7c1f460988c34f5a0efb3459f3d9f04b59d2daf53a6dc83` (897,030 bytes, 313 files).

Published evidence contains this summary and the implementation/regression patch. Full execution artifacts remain in the private case archive.
