# precision-skill

[English](README.md) | 简体中文

用于不同硬件与软件执行栈之间推理、SFT、LoRA 和 RL 数值比较的 Agent Skill。

## 使用

将 [skills/precision-skill](skills/precision-skill/) 复制到客户端的 skill 目录。提供任务和已有产物，或采集所需的模型、数据与平台连接方式。agent 负责准备环境、接入原生执行路径、编写缺少的采集与比较代码并运行任务。在 skill 开发阶段，将可复用的实现整理进工具库。

例如：

- 比较两份执行 manifest 中的梯度和实际权重更新。
- 使用相同 batch 和初始适配器状态，在两个执行栈上采集 LoRA SFT 单步并比较。

[SKILL.md](skills/precision-skill/SKILL.md) 组织采集与比较流程。`references/` 中的三个任务文档覆盖推理、SFT（全参 / LoRA）和 RL，说明采集内容、原生框架接入位置（含 Other/custom）及应报告的指标。`comparison.md` 定义坐标配对、通用指标和工具用法；分布指标在推理文档中定义。训练采集阶段在 SFT 中定义，由 RL 复用。

结果包含实际配置、指标值及聚合方式、覆盖数量、最大误差坐标和产物路径。任务决定对等执行配置之间的共同条件及各侧允许存在的差异；比较指标采用对称定义。采集保留原生算子并记录各侧依赖。

需要层间比较时，agent 先确认范围与资源上限，再自主接入采集工具并细化采样。[层间比较指南](skills/precision-skill/references/layerwise.md)包含 msProbe 的能力、接入与试采集步骤，其他执行路径可使用原生 hook。

## 典型案例

### Ascend 910C（昇腾）NPU 与 NVIDIA H200 GPU

使用 Qwen3.5-9B-Base 和各平台原生软件栈，已完成以下双端采集与数值比较：

| 框架 | 已完成的比较 |
|---|---|
| PyTorch / Transformers；LoRA 使用 PEFT | 完整前向推理、全参 SFT 和 LoRA SFT |
| SGLang | 原生缓存推理，固定续写 token；比较 prompt 评分与约束前分布 |
| TRL | 固定轨迹 GRPO 的动作评分、loss、梯度及单步参数更新 |
| verl / FSDP2 | 固定轨迹 GRPO 的动作评分、loss、梯度及单步参数更新 |

## 工具与开发

skill 组织任务，任务适配代码接入原生执行路径，[工具库](skills/precision-skill/references/comparison.md#tools)提供可复用的操作。`scripts/` 包含命令行工具、可通过 Python 调用的工具及内部公共模块。工具 docstring 定义输入输出，任务文档负责选择采集边界和指标。

工具负责检查运行环境初始化和实际导入的接口、采集固定历史的推理分布、规划内部采样、导出和校验产物，以及比较分布、张量观察值（含激活）、梯度、更新和目标评分。msProbe 是可选的采集后端；内置规划工具生成有界的 L0 采集范围并估算张量数据量，任务适配代码负责执行采集。缺少的任务接入按需实现，开发阶段再将可复用操作整理进工具库。数组工具依赖 NumPy；目标评分工具使用标准库。

[证据工具](skills/precision-skill/references/comparison.md#portable-evidence)为选定产物记录原始/导出哈希，并通过已安装 skill 的命令行工具重放保留的比较。

完整采集任务及 agent 的预期执行过程见[固定历史推理示例](skills/precision-skill/examples/fixed-history-inference.md)，涵盖环境准备、采集接入、校验、比较和交付。

使用[可运行的梯度比较示例](skills/precision-skill/examples/compare_gradients.py)，比较同一小型分类模型、同一 batch 下的 FP32 与 BF16 梯度。在 skill 目录运行，依赖 Torch 和 NumPy：

```sh
python3 examples/compare_gradients.py
```

示例在 CPU 上实际执行两次前向和反向，随后输出 loss、梯度差异指标和产物目录。

仓库组织与测试命令见 [AGENTS.md](AGENTS.md)。
