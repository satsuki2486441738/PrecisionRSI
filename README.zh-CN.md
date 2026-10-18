# precision-skill

[English](README.md) | 简体中文

用于不同硬件与软件执行栈之间推理、SFT、LoRA 和 RL 数值比较的 Agent Skill。

## 使用

将 [skills/precision-skill](skills/precision-skill/) 复制到客户端的 skill 目录。提供任务和已有产物，或采集所需的模型、数据与平台连接方式。

例如：

- 比较两份执行 manifest 中的梯度和实际权重更新。
- 使用相同 batch 和初始适配器状态，在两个执行栈上采集 LoRA SFT 单步并比较。

[SKILL.md](skills/precision-skill/SKILL.md) 按任务选择路径：已有产物直接使用比较器；需要采集时读取所选任务协议。`references/tasks/` 按推理、SFT、RL 组织采集要求、框架接入（含 Other/custom）和应报告的指标，SFT 包含全参微调与 LoRA；`references/methods/` 存放按需读取的采集、训练状态、指标和工具方法。

结果包含配置、主要差异、比较坐标和最大差异位置。任务选择对等的执行配置，可使用相同或不同硬件；比较指标采用对称定义。采集保留各侧原生算子和固定的依赖版本。

## 工具与开发

[离线工具](skills/precision-skill/references/methods/tools.md)比较分布、梯度、更新和目标评分。产物工具依赖 NumPy；cache-off 目标评分比较器使用标准库。

仓库组织与测试命令见 [AGENTS.md](AGENTS.md)。
