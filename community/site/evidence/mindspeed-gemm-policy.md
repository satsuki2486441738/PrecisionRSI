# 修复后追踪：同输入 GEMM 与 cuBLAS 计算模式

模型为 Qwen3.5-9B-Base。固定两组此前采集的归一化张量、同一投影权重分片，在 H200 与昇腾 910C 上执行原生 linear。各侧默认计算逐值复现原模型的首次投影；10 个策略配置分别重复 3 次，输出均保持一致。

## 同输入比较

每组输出包含 586,720 个值。使用 GPU 侧归一化张量作为两端共同输入时，原生 BF16 跨端 RMSE 为 **0.0004351994452756103**，1,005 个值不同。改用 NPU 侧归一化张量时为 **0.0004212083355701658**，958 个值不同。输入和权重的 SHA256 均保留在 JSON。

关闭 CUDA 的 `allow_bf16_reduced_precision_reduction` 后，两组输出均完全不变。改为 FP32 输入会降低 GPU 相对参照的误差，NPU 则没有同样改善。

## 进一步隔离 cuBLAS

GPU 环境为 PyTorch 2.9.0+cu126、cuBLAS 12.6.4.1。直接使用独立 cuBLAS handle 计算相同 BF16 操作数。以下指标统一使用 GPU 侧归一化输入，参照为 CPU FP64 运算后直接 RNE 舍入到 BF16。

| GPU 路径 | 对参照的 RMSE | 不同值数量 |
| --- | ---: | ---: |
| Torch 原生 / cuBLAS 默认 | 0.00043448739994732 | 975 |
| cuBLAS 禁止低精度归约 | 0.00043448739994732 | 975 |
| cuBLAS 输出 FP32 后转 BF16 | 0.00043448739994732 | 975 |
| cuBLAS pedantic | 0.00013881869221573221 | 154 |
| Torch FP32 输入 | 0.00007302994176884011 | 143 |

默认 cuBLAS 与 Torch 逐值一致。Profiler 将默认调用对应到 `nvjet_tst_80x64_64x11_4x1_v_bz_TNN`，pedantic 对应到 `magma_sgemmEx_kernel<float, __nv_bfloat16, ...>`。pedantic 同时改变计算类型、math mode 与 kernel 选择，本轮定位到这一计算策略组合。

单次 Profiler 中，默认 kernel 约 11.7 μs，pedantic 约 559 μs。这是一次轨迹观察，尚未作为性能基准，也未部署为模型的新数值策略。

## 独立参照检查

使用 CPU `math.fsum` 对 BF16 操作数的精确 FP64 乘积独立求和，两组输入分别检查 1,088 和 1,046 个关键坐标，与原 FP64/BF16 参照全部一致。GPU 原生结果有 36 个坐标偏离参照超过一个 BF16 可表示间隔，最多 18 个间隔，出现在接近零的输出；因此 GEMM 差异并非都只是最终舍入中点的选择。

本轮确认了同输入下计算路径对误差的影响，尚未确认新增库语义缺陷。整模型保留上轮配置；如采用新的数值策略，后续需验证原任务前向、反向及计算成本。

NPU 本轮用时 31.054 秒，累计 **747.100 / 900 秒**。本轮作业已结束并清理；原始采样、Profiler trace、代码及独立求和数据留在私有案例中。

官方接口说明：[PyTorch 2.9 数值精度](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms)、[CUDA 12.6 cuBLAS 计算类型](https://docs.nvidia.com/cuda/archive/12.6.3/cublas/index.html#cublascomputetype-t)。
