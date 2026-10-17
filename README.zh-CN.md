# precision-skill

[English](README.md) | 简体中文

用于比较 NVIDIA GPU 与昇腾 NPU 执行栈数值结果的 [Agent Skill](https://agentskills.io/home)。

## 使用

将 [skills/precision-skill](skills/precision-skill/) 整个目录复制到客户端的 skill 位置，保留目录名。提供任务、双端执行栈和已有产物；agent 复用现有配置与授权，对齐条件、采集数据、验证覆盖范围并报告数值差异。

| 任务 | 方法 |
|---|---|
| [推理](skills/precision-skill/references/inference.md) | 比较 next-token top-100 并集及合并尾部：KL、Top-1 一致率、概率误差和覆盖率。 |
| [SFT](skills/precision-skill/references/training.md#sft) | 同状态单步：全部有效目标 token 评分、全量梯度和全量实际参数更新；LoRA 适配器范围与冻结基座证据；初始／恢复 optimizer 状态基线及回归复用。 |
| [RL](skills/precision-skill/references/training.md#rl) | 固定轨迹，比较算法所需依赖、目标函数、梯度和更新量。 |

以 GPU 为参考。未预设阈值的数值诊断不代表精度通过。只读取当前任务需要的参考；案例配置、调研依据和执行历史留在方法说明之外。

按当前采集层选择集成参考：

| 层次 | 参考 |
|---|---|
| 推理引擎 | [vLLM / SGLang](skills/precision-skill/references/engines.md)：prefill、固定历史的缓存 decode 与评分语义。 |
| RL 框架 | [TRL / verl](skills/precision-skill/references/rl-frameworks.md)：轨迹准备、原生目标函数与更新边界。 |
| 训练后端 | [FSDP / ZeRO / Megatron-Core / MindSpeed](skills/precision-skill/references/training-backends.md)：原生存储、optimizer 阶段与逻辑张量映射。 |

采集器和执行证据由案例保存。这些参考描述方法；各案例记录实际覆盖范围和验证状态。

## 工具与组织

[离线工具](skills/precision-skill/references/tools.md)可校验产物，并基于 NumPy 分块比较 top-100 分布、全量梯度和实际更新；原有标准库目标评分比较器继续保留。采集器、格式适配和任务 loss 重建仍放在案例中；自比较不代表跨设备精度一致。

- [SKILL.md](skills/precision-skill/SKILL.md)：pipeline 与参考入口。
- [比较](skills/precision-skill/references/comparison.md)：对齐规则、公式和结果解释。
- [环境](skills/precision-skill/references/environment.md)：共同版本与执行关联证据。
- [工作流](skills/precision-skill/references/workflow.md)：设备运行、重试和交接。
- [架构](skills/precision-skill/references/architecture.md)：采集器职责与案例布局。

私有 `projects/` 保存案例配置、采集器与运行结果；权重、数据和环境实体保留在资源目录。skill 技术文档使用英文，根目录说明保持中英双语。

使用含 NumPy 的现有 Python 环境，在仓库根目录运行维护测试：`python -B -m unittest discover -s tests -v`。
