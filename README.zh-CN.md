# precision-skill

[English](README.md) | 简体中文

用于 GPU/NPU 推理、SFT、LoRA 和 RL 数值比较的 Agent Skill。

## 使用

将 [skills/precision-skill](skills/precision-skill/) 复制到客户端的 skill 目录。提供任务和已有产物，或采集所需的模型、数据与平台连接方式。

例如：

- 比较这两份 GPU/NPU manifest 中的梯度和实际权重更新。
- 使用相同 batch 和初始适配器状态，在双端采集 LoRA SFT 单步并比较。

[SKILL.md](skills/precision-skill/SKILL.md) 按任务选择路径：已有产物直接使用比较器；需要采集时读取所选任务协议。框架接入和指标定义按需读取。

结果包含配置、主要差异、比较坐标和最大差异位置。GPU 为参考；采集保留两端原生算子和固定的依赖版本。

## 工具与开发

[离线工具](skills/precision-skill/references/tools.md)比较分布、梯度、更新和目标评分。产物工具依赖 NumPy；cache-off 目标评分比较器使用标准库。

仓库组织与测试命令见 [AGENTS.md](AGENTS.md)。
