# 历史缺陷回移植：PEFT LoRA SVD 重复缩放

2026-10-22 · PEFT 0.17.1 · 原生微型 CPU Linear。此条复现上游已修复的真实缺陷，并回移植一行补丁；不是新发现。

## 根因与独立公式

源适配器的有效增量为 `s_i B_i A_i`，其中 `s_i = alpha_i / r_i`。无截断 SVD 应保持 `Σ w_i s_i B_i A_i`。原实现调用已包含缩放的 `get_delta_weight` 后，又乘以 `s_i`，实际组合成 `Σ w_i s_i² B_i A_i`。补丁仅去掉第二次缩放，不改 dtype、SVD、rank 或接口。

独立 FP64 oracle 从保存的源因子与配置直接构造 dense affine map，不调用 PEFT delta/scaling helper；对平方输出求导验证输入梯度。零输入及全部基向量覆盖每个微型仿射映射系数。

## 同负载验证

24 个原生用例：FP32/FP64 × 三种形状（4×4、3×5、5×3）× 单位/不等 source scale × 正/有符号权重。两个源适配器 rank=2；SVD 输出 rank 为 `min(in,out)`，排除预期截断误差。测试和全部 24 组输入/源状态 fixture 哈希在修复前后一致，源因子无修改。

| 边界 | 每侧配对标量数 | 最大绝对误差：修复前 | 修复后 |
| --- | ---: | ---: | ---: |
| delta | 368 | 1.8386437858590967 | 1.9718805555068997e-07 |
| output | 672 | 4.5465921511820895 | 4.144419198937044e-07 |
| basis | 464 | 1.8386438626891457 | 1.6738573316299465e-07 |
| input_gradient | 672 | 56.05958776139278 | 2.5047252929510933e-06 |
| checkpoint_output | 672 | 4.5465921511820895 | 2.847181863607773e-07 |
| merged_output | 672 | 4.5465921511820895 | 3.3760613910027715e-07 |

共 3,520 个配对标量观测，包含相关的前向、保存加载和合并观测，并非 3,520 次独立试验。原始源码 12 个非单位缩放用例失败、12 个单位缩放对照通过；修复后 24/24 全部通过，无执行错误。原始两适配器负载也再次执行，delta entry 从 2.2812502384 恢复为预期的 0.78125。

FP32 容差 atol=rtol=1e-5，FP64 为 1e-10。原 FP64 错误同样存在，支持这是代数语义问题。checkpoint 使用原生 PyTorch 序列化；merge 使用 `safe_merge=True`。

## 固定来源与范围

- 历史版本：`53c25fe4fdd7c6aa4b40db0560815ea570d32303`。
- 上游修复：[`9a267b211fa6b43ba403e255b5e0dd59bf7c982e`](https://github.com/huggingface/peft/commit/9a267b211fa6b43ba403e255b5e0dd59bf7c982e)，PR #3111。
- 上游合并提交：`2659fcb52e23d0f8c526067bdb64025ce71c71ba`。
- 已检查的上游快照：`99dfc7021c8c743780502713621d7666c4009a63`；重复因子已移除。缺陷引入提交未做二分定位。

范围仅为原生微型 CPU Linear；未验证 Qwen3.5-9B、GPU/NPU、分布式、BF16、量化、优化器步骤或 LoRA 因子反向。不含截断 SVD、随机合并、embedding/convolution、safetensors 往返或 unmerge。保存加载检查用于确认组合结果持久性，不另计一项 checkpoint 修复。

[结构化指标、版本和证据哈希](peft-lora-svd-scaling.json) · [最小回移植补丁](peft-lora-svd-scaling.patch)。完整执行产物和永久回归留在私有 case。用户验收尚未判定。
