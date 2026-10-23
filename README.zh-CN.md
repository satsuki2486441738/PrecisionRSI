# Precision-Skills: Numerical Precision Comparison and Repair

[English](README.md) | 简体中文

让 agent 完成不同硬件与执行环境中的框架适配、数值精度比较与修复。

## 使用

将 `framework-adapt`、`precision-compare` 和 `precision-repair` 并排安装到客户端的 skill 目录，执行时按任务调用所需组件：

| Skill | 任务与交付 |
|---|---|
| [Framework Adapt](skills/framework-adapt/SKILL.md) | 为目标模型、硬件和任务部署或适配框架，交付可复现的执行入口与验证证据 |
| [Precision Compare](skills/precision-compare/SKILL.md) | 采集与比较数值结果，输出指标、覆盖率和证据 |
| [Precision Repair](skills/precision-repair/SKILL.md) | 确定数值问题的原因、修复实现，并回到原始任务复验 |
| [Operator Optimize](skills/operator-optimize/SKILL.md) | 占位，尚未实现 |

三个包共用工具和组件间链接；agent 按任务加载相关 skill 与参考文档。`precision-compare` 是原 `precision-skill` 比较包的新名称，更新时替换原安装目录。

对于端到端任务，提供模型工作负载、目标执行栈和资源范围。agent 补齐原生执行能力，比较指定的数值结果，并在任务包含修复时处理确认的问题。三个 skill 共用[任务执行与资源目录](skills/precision-compare/references/comparison.md#shared-tasks)：复用有效的环境准备和采集，失败时定位下一组件，代码变化后重跑受影响的验证。各组件也可单独承担对应任务。

进行比较时，提供任务和已有产物，或采集所需的模型、数据与平台连接方式。agent 负责准备环境、接入原生执行路径、编写缺少的采集与比较代码并运行任务。在 skill 开发阶段，将可复用的实现整理进工具库。

例如：

- 比较两份执行 manifest 中的梯度和实际权重更新。
- 在提供的执行栈上运行这个模型的 SFT 负载，比较数值结果，修复确认的正确性问题，再复验原始任务。

[SKILL.md](skills/precision-compare/SKILL.md) 组织采集与比较流程。`references/` 中的三个任务文档覆盖推理、SFT（全参 / LoRA）和 RL，说明采集内容、原生框架接入位置（含 Other/custom）及应报告的指标。`comparison.md` 定义坐标配对、通用指标和工具用法；分布指标在推理文档中定义。训练采集阶段在 SFT 中定义，由 RL 复用。

已有对应关系的数组可直接使用 [compare_arrays](skills/precision-compare/scripts/array_compare.py)，输出精确相等、覆盖情况和有界的差异定位；误差指标按需计算。需要新增采集时再读取任务文档，需要映射或执行校验时使用相应 artifact 接口。

结果包含任务要求的比较结果、实际配置、覆盖数量、差异坐标和复现产物。任务决定对等执行配置之间的共同条件及各侧允许存在的差异；比较指标采用对称定义。采集保留原生算子并记录各侧依赖。

需要层间比较时，agent 先确认范围与资源上限，再自主接入采集工具并细化采样。[层间比较指南](skills/precision-compare/references/layerwise.md)包含 msProbe 的能力、接入与试采集步骤，其他执行路径可使用原生 hook。

## 典型案例

### Ascend 910C（昇腾）NPU 与 NVIDIA H200 GPU

使用 Qwen3.5-9B-Base 和各平台原生软件栈，已完成以下双端采集与数值比较：

| 框架 | 已完成的比较 |
|---|---|
| PyTorch / Transformers；LoRA 使用 PEFT | 完整前向推理、全参 SFT 和 LoRA SFT |
| Transformers Trainer | 原生全参 SFT 单步（抽样梯度/更新）；单设备及两卡 FSDP2 LoRA SFT 梯度累积、两步 AdamW 与 checkpoint 恢复（完整 adapter 梯度、更新和优化器状态）；目标评分 |
| vLLM / vLLM-Ascend | 原生 prefill 目标评分；固定续写的缓存推理分布与目标评分 |
| SGLang | 原生缓存推理，固定续写 token；比较 prompt 评分与约束前分布 |
| TRL | 原生 SFTTrainer LoRA SFT：completion-only 标签、梯度累积、两步 AdamW 及原生 checkpoint 恢复（完整 adapter 梯度、更新及优化器状态）；固定轨迹 GRPO 的动作评分、loss、梯度及单步参数更新 |
| verl / FSDP2 | 固定轨迹 GRPO 的动作评分、loss、梯度及单步参数更新 |

### 精度修复

在固定的历史软件栈上，修复 Qwen3.5-9B-Base 的零中心 RMSNorm 权重语义。验证覆盖 NPU 原生算子的前向/反向、权重保存与加载，以及加载权重字节不变时的原始 TP4 模型前向复验。

另用小型原生 CPU 工作负载验证了以下历史缺陷的回补：

| 库 / 固定版本 | 修复内容 | 验证范围 |
|---|---|---|
| TRL 0.7.10 | IPO 的 completion logprob 平均归约 | 原生 DPOTrainer loss、全部梯度和一次 SGD 更新；6 项回归 |
| PEFT 0.17.1 | SVD adapter 合并重复应用 LoRA 缩放 | 前向、输入梯度、checkpoint 加载及合并；24 组工作负载 |
| Transformers 4.51.3 | Qwen2 attention mask 忽略关闭滑动窗口的配置 | Eager/SDPA 完整前向及动态/静态 cache；18 组工作负载和 16 组 mask 检查 |

[修复历史](https://satsuki2486441738.github.io/precision-skills/)记录受影响版本、上游修复版本、补丁、前后指标和验证范围。


## 工具与开发

三个 skill 共同组织一个工作负载，任务适配器接入原生执行，[工具库](skills/precision-compare/references/comparison.md#tools)执行并复用各阶段。`scripts/` 包含命令行工具、可通过 Python 调用的工具及内部公共模块。工具 docstring 定义输入输出，`tool_help.py` 可在不导入依赖的情况下查询选定接口。任务文档负责选择采集边界和指标。

工具负责检查运行环境初始化和实际导入的接口、记录命令执行、采集固定历史的推理分布、规划内部采样、导出和校验产物，以及比较分布、张量观察值（含激活）、梯度、更新和目标评分。张量诊断还可统计 NaN／Inf 及其位置，单独报告双侧有限值的配对覆盖率和差异指标。msProbe 是可选的采集后端；内置规划工具生成有界的 L0 采集范围并估算张量数据量，任务适配代码负责执行采集。缺少的任务接入按需实现，开发阶段再将可复用操作整理进工具库。嵌套状态比较复用数组指标，同时检查 checkpoint 的结构与元数据。数组工具依赖 NumPy；目标评分工具使用标准库。

[证据工具](skills/precision-compare/references/comparison.md#portable-evidence)为选定产物记录原始/导出哈希，并通过已安装 skill 的命令行工具重放保留的比较。

完整采集任务及 agent 的预期执行过程见[固定历史推理示例](skills/precision-compare/examples/fixed-history-inference.md)，涵盖环境准备、采集接入、校验、比较和交付。

使用[可运行的梯度比较示例](skills/precision-compare/examples/compare_gradients.py)，比较同一小型分类模型、同一 batch 下的 FP32 与 BF16 梯度。在 skill 目录运行，依赖 Torch 和 NumPy：

```sh
python3 examples/compare_gradients.py
```

示例在 CPU 上实际执行两次前向和反向，直接比较已对应的梯度数组。每次运行保留两份 NPZ、实际配置和比较报告，随后输出 loss、指标和产物目录。

在 skill 目录运行[原生端到端示例](skills/precision-compare/examples/workflow_demo.py)：

```sh
python3 examples/workflow_demo.py --output-root /tmp/precision-workloads
```

示例执行两步 Torch SFT 负载，检测一个受控的目标函数错误，应用已知修正，再复验原始训练轨迹并复用环境准备。最后一次重放不启动任何原生命令。

仓库组织与测试命令见 [AGENTS.md](AGENTS.md)。

## 评测与修复历史

[历史网页](https://satsuki2486441738.github.io/precision-skills/)包含两个标签页：**精度评测**展示推理 / SFT / RL 的比较配置、数值指标、覆盖率和证据；**精度修复**保留按库分组的修复、问题归因及前后验证。记录及公开证据维护在 [site/](site/README.md)。
