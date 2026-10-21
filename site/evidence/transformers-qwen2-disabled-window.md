# 历史缺陷回移植：Qwen2 关闭滑窗时的 causal mask

2026-10-22 · Transformers 4.51.3 · 原生微型随机 Qwen2 CPU FP32。此条复现上游已修复的真实缺陷并回移植一行 guard；不是新发现，也不是预训练模型或完整 9B 验证。

## 根因与独立 oracle

`Qwen2Config` 在 `use_sliding_window=False` 时仍保留数值 `sliding_window`。原 mask builder 只检查该数值，错误屏蔽合法的历史 key。补丁增加 `config.use_sliding_window` 条件，不改变接口、dtype 或注意力实现。

独立 oracle 按 `k <= q` 及非 padding key 构造显式 4D causal mask，交给同一原生模型以绕过有缺陷的 mask builder；启用滑窗的对照独立增加 `q-k < window`。不调用被修复的 guard 或 mask helper。

## 同负载验证

随机初始化 24,928 参数 Qwen2：2 层、hidden=32、4 query heads / 2 KV heads、词表 97、batch=2、长度 8、seed=23。CPU FP32、单线程、eval/inference、确定性算法。eager/SDPA × 无/左/右 padding × 完整前向/DynamicCache/StaticCache，共 18 个场景。cache prefill=5，随后逐 token 解码 3 次；StaticCache 容量 12。

| 对独立 oracle 的测量 | 修复前 | 修复后 |
| --- | ---: | ---: |
| Logit 最大绝对误差 | 0.12348369508981705 | 5.960464477539063e-08 |
| 全局 logit RMSE | 0.018204872510839642 | 6.498900616537927e-09 |
| Target logprob MAE（nat/token） | 0.006720593988269109 | 1.7546466774669461e-09 |
| Target 加权 NLL 绝对差异（nat/token） | 0.0011539648998306618 | 1.9100365733493163e-10 |
| 精确 mask 不同元素数 | 27 | 0 |

覆盖 25,608 个有效 logit、228 个 target scores，另有 16 个 mask 用例、414 个 mask 元素。坐标包含执行场景标识，跨场景重复坐标按观测计数。排除 padding query，target t 配对 t-1 logits 且两个端点均非 padding。修复前后模型状态、配置、输入和 position IDs 相同。

永久回归原版失败 22 个子用例（18 个原生场景、4 个禁用滑窗 mask 场景），12 个对照通过；补丁版 34/34 通过，两端均无执行错误。logit 容差 atol=2e-6、rtol=0，mask 必须零差异。

## 固定来源与范围

- 历史版本：`5f4ecf2d9f867a1255131d2461d75793c0cf1db2`。
- 引入配置变更的 refactor：`96f01a36ac60c57be279cb7f027eac7e37fa90e8`。
- 上游实际修复：[`0a52bd24031001a887a63aa6df2bd61785ec4fb3`](https://github.com/huggingface/transformers/commit/0a52bd24031001a887a63aa6df2bd61785ec4fb3)，PR #38045。
- 已验证修复的 4.52.0 提交：`113424bcd53b92600f77d82f48add0a60fb41556`；未穷尽搜索首个修复版本。

仅验证上述 CPU FP32 微型原生推理；未验证预训练 Qwen、9B、生成质量、训练/反向、GPU/NPU、BF16、长上下文、FlashAttention2、SlidingWindowCache 或其他架构。仅在隔离执行中修改生成的 modeling 文件，未进行上游源码再生成或提交 PR。

[结构化指标、来源及证据哈希](transformers-qwen2-disabled-window.json) · [最小回移植补丁](transformers-qwen2-disabled-window.patch)。完整产物和永久回归留在私有 case。用户验收尚未判定。
