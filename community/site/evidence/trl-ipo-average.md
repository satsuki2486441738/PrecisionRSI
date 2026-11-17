# 历史缺陷回移植：TRL IPO 的 completion 均值

2026-10-22 · TRL 0.7.10 · 原生微型 CPU GPT-2。此条复现上游已修复的真实缺陷，并将最小补丁回移植到历史版本；不是新发现。

## 根因与修复

原生 `DPOTrainer.concatenated_forward` 对 IPO 也无条件累加 completion logprob。预期实现对每条 completion 取 token 均值，避免长度改变 IPO gap。仅将 `average_log_prob=False` 改为 `average_log_prob=self.loss_type == "ipo"`；sigmoid/hinge 保持原行为。

预期语义来自上游修复及作者确认，不声称论文正文独立规定了该归一化。独立 oracle 对 chosen/rejected 分别前向，按显式 causal label 坐标计算 logsumexp，含 EOS、排除 prompt/padding，不调用 TRL 的评分、拼接或 loss helper。

## 同负载验证

正常构造并运行 DPOTrainer 的 tokenization、collation、compute_loss、backward 和一次 SGD（lr=0.01）。随机初始化 GPT-2：1 层、宽 8、2 头、词表 12、1,240 个参数；两对不等长偏好，共 13 个 completion targets。修复前后输入与初始参数哈希相同。

| FP64 与独立 oracle 的误差 | 修复前 | 修复后 |
| --- | ---: | ---: |
| Loss 绝对误差 | 0.5565106547209888 | 8.881784197001252e-16 |
| 参数梯度最大绝对误差 | 4.726013461672909 | 8.881784197001252e-16 |
| 实际 SGD 更新最大绝对误差 | 0.04726013461672908 | 1.0408340855860843e-17 |

原始永久回归：4 个 IPO 测试失败，2 个 sigmoid/hinge 对照通过；同一测试修复后 6/6 通过。覆盖 FP32/FP64、不等长 completion、额外右侧 padding、单对偏好及全部参数梯度与存储更新。FP64 容差 1e-10，FP32 容差 2e-5，允许独立公式及分别/拼接前向的运算重排。

## 固定来源与范围

- 原版本提交：`09ca7607d50714a80c9f0feb206b29f1c5a49bab`。
- 原 IPO 功能提交：`55d7c952c796345d9fa520e692f7760372b14b43`。
- 上游修复：[`29d439a2043edf4455b05cae5a1e2ade69d22794`](https://github.com/huggingface/trl/commit/29d439a2043edf4455b05cae5a1e2ade69d22794)，PR #1265；该问题已在上游修复。
- 已检查的上游快照：`770e1be9c0d01bdfbf50bf1281d75a8ef341b474`。

验证范围仅为上述原生微型 CPU 负载，不含完整 Trainer.train、收敛、Qwen3.5-9B、GPU/NPU、分布式、BF16/FP16、参考缓存、encoder-decoder 或空 completion。补丁未改变 dtype 或数值策略。

[结构化指标、版本和证据哈希](trl-ipo-average.json) · [最小回移植补丁](trl-ipo-average.patch)。完整执行产物和永久回归留在私有 case。用户验收尚未判定。
