# precision-skill

[English](README.md) | 简体中文

用于比较 NVIDIA GPU 与昇腾 NPU 执行栈数值结果的 [Agent Skill](https://agentskills.io/home)。

## 使用

将 [skills/precision-skill](skills/precision-skill/) 整个目录复制到客户端的 skill 位置，保留目录名。提供任务、双端执行栈和已有产物；agent 复用现有配置与授权，对齐条件、采集数据、验证覆盖范围并报告数值差异。

| 任务 | 方法 |
|---|---|
| [推理](skills/precision-skill/references/inference.md) | 比较 next-token top-100 并集及合并尾部：KL、Top-1 一致率、概率误差和覆盖率。 |
| [SFT](skills/precision-skill/references/training.md#sft) | 同状态单步：全部有效目标 token 评分、全量梯度和全量实际参数更新。 |
| [RL](skills/precision-skill/references/training.md#rl) | 固定轨迹，比较算法所需依赖、目标函数、梯度和更新量。 |

以 GPU 为参考。未预设阈值的数值诊断不代表精度通过。只读取当前任务需要的参考；案例配置、调研依据和执行历史留在方法说明之外。

vLLM 和 SGLang 使用[引擎采集指南](skills/precision-skill/references/engines.md)，覆盖原生 prefill、固定历史的缓存 decode、原始评分语义及完整分布采集。引擎适配器仍由案例实现；方法说明不代表设备验证已完成。

TRL 或 verl 的 RL 任务使用[框架指南](skills/precision-skill/references/rl-frameworks.md)，明确固定轨迹回放和原生 optimizer step 的采集边界。默认指标覆盖动作评分、loss、梯度及实际更新；算法专属诊断按需展开。框架采集器仍由案例实现。

## 工具与组织

随包[比较器](skills/precision-skill/scripts/compare_scores.py)实现 schema-v2 full-forward **目标 token** 的 NLL、ΔNLL、MAE 和 P99，阈值为 null。Top-100、训练和 cache 比较使用案例专用工具；文档定义不代表随包比较器已实现这些能力。

- [SKILL.md](skills/precision-skill/SKILL.md)：pipeline 与参考入口。
- [比较](skills/precision-skill/references/comparison.md)：对齐规则、公式和结果解释。
- [环境](skills/precision-skill/references/environment.md)：共同版本与执行关联证据。
- [工作流](skills/precision-skill/references/workflow.md)：设备运行、重试和交接。
- [架构](skills/precision-skill/references/architecture.md)：采集器职责与案例布局。

私有 `projects/` 保存案例配置、采集器与运行结果；权重、数据和环境实体保留在资源目录。skill 技术文档使用英文，根目录说明保持中英双语。

在仓库根目录运行维护测试：`python -B -m unittest discover -s tests -v`。
