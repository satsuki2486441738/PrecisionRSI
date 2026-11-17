# NumericBridge: Numerical Precision Comparison and Repair

[English](README.md) | 简体中文

NumericBridge 是面向 RL 的精度对比诊断与修复优化框架，包含可复用 skills、可选研究编排和公开社区记录。

## 架构

| 层 | 职责 |
|---|---|
| [Skills](skills/precision-compare/SKILL.md) | 负载适配、精度比较、诊断修复与原任务复验 |
| [precision_core](precision_core/README.md) | 公共执行、资源与运行时身份、证据工具 |
| [precision_community](precision_community/README.md) | 公开记录校验、同侧配置索引、绑定证据的显式关系 |
| [precision_rsi](precision_rsi/README.md) | 可选的 Agent 研究、预算、候选评估与激活 |
| [precision_maintenance](precision_maintenance/README.md) | 数据维护、公开导出、安装及 CI 命令 |
| site/ | 基于已审查公开记录的静态社区页面 |

RSI 依赖公共核心；社区记录不依赖 RSI。检索用于选择候选方案，实际组合的验证由任务负责。公开记录保留范围和不可变证据，完整轨迹与大数组留在本地。

## 方法

NumericBridge 包含两个闭环：完成模型工作负载，再将执行经验用于改进后续任务。

**任务 loop**

```text
配环境 → 复现库 → 精度检验 → 精度修复 → 算子优化
```

“复现库”指复现目标库的原生模型工作负载。从首个缺失阶段进入，达到任务要求后结束。精度修复与算子修改后，返回原始工作负载做精度检验；算子优化同时验证性能收益。

**优化 loop**

```text
任务轨迹 → 选择改进对象 → 构建候选更新 → 独立评测 → 下一系统版本
```

| 迭代对象 | 产物与复用方式 |
|---|---|
| 分析工具 | 经测试的采集器、诊断工具、适配器和通用轮子，提供可调用接口 |
| 上下文记忆 | 带适用条件的简短规则，按任务需要检索 |
| 训练数据 | 标注执行结果的轨迹、失败—修复对和回归样本，去重并按任务族划分数据集 |
| 任务 loop 流程 | 版本化的 skill 指令、阶段路由、工具选择、执行复用和验证范围 |

成功、失败和开销过高的任务轨迹都进入改进过程。改进 agent 从中选择重复出现的瓶颈或缺失能力，构建预算内的候选更新；有新证据时再触发，无需每个任务都额外调用模型。任务达到要求后结束，迭代 loop 则在独立预算内跨任务持续。

在独立任务上比较候选系统版本的正确性、完成时间和成本，计入更新的构建与维护开销，并保持任务目标、正确性标准和评测覆盖范围一致。有效版本供后续任务使用；预算耗尽或未观察到收益时暂停改进。训练数据先校验来源、结果标注、去重和数据集划分；后续权重训练与评测使用通过检查的数据集。

封闭式任务（精度检验／修复）评估正确完成率、时间和成本。开放式任务（算子优化）评估给定预算下达到的最优有效性能，以及性能随预算增长的变化。

[系统入口](precision_rsi/README.md)将原生工作流接入现有 agent harness。独立改进阶段根据任务证据构建工具、记忆、数据集和指令／路由更新；运行时更新组成版本化的组合，评测后供后续任务使用，数据集则通过检查后保留任务结果标注。[研究快照](precision_rsi/README.md#research-snapshots)记录轮次和候选边界，支持完整性校验、独立目录恢复和已准入版本回滚。在仓库目录运行：

```sh
python -m precision_rsi run task.json --workspace projects/my-task/session
python -m precision_rsi iterate series.json --workspace projects/my-stream/session
```

任务队列可在总时间预算内，根据新证据触发改进。算子优化与 agent 权重训练为后续扩展。

## 使用

[项目架构与四个整理边界](.github/project-architecture.md)说明当前实现与后续方向。

[RL 精度诊断与修复方案（草案）](.github/rl-precision-workflow-proposal.md)基于本仓库已有能力，约定可复用的案例执行与交付流程。

将 skills 导出到新的客户端 skill 目录；导出器会带上公共执行包，使 Compare 能脱离仓库运行：

```sh
python -B -m precision_maintenance install --output /path/to/client/skills
```

更新方式见[独立安装说明](precision_core/README.md#portable-skills)。执行时根据下一项缺失能力加载组件：

| Skill | 进入条件与交付 |
|---|---|
| [Framework Adapt](skills/framework-adapt/SKILL.md) | 缺少原生执行能力时，部署或适配目标负载并验证执行 |
| [Precision Compare](skills/precision-compare/SKILL.md) | 缺少采集或比较结果时，输出数值指标、覆盖率和证据 |
| [Precision Repair](skills/precision-repair/SKILL.md) | 任务要求诊断或修复时，建立正确性证据、修复确认的缺陷并复验原始负载 |
| [Operator Optimize](skills/operator-optimize/SKILL.md) | 占位，尚未实现 |

从已有的执行命令、输入／状态、采集器和结果开始，只为缺失的工作加载相应组件与参考文档；共享工具可直接调用，无需加载所属 skill。`precision-compare` 是原 `precision-skill` 比较包的新名称，更新时替换原安装目录。

提供期望结果、工作负载或已有产物、目标执行栈和资源范围。需要其他能力时，在同一工作目录继续，跨组件复用执行入口、输入／状态、采集器和结果。[执行工具与资源目录](skills/precision-compare/references/comparison.md#shared-tasks)支持复用命令链。修复由任务选择；代码变化后复验受影响的范围。

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

### Qwen3.8-Flash-Next

[文本精度报告](.github/qwen38-precision-report.md)和[已核验证据](site/evidence/qwen38-precision-20261106.json)覆盖 26 项原生核心采集、完整 32K 推理、LoRA、实际 Trainer/SFTTrainer、固定轨迹 TRL/verl 组件及 MoE/QSA 干预。最终推理比较 GPU 自动分布与 NPU TP8，其余链路保留各自实际配置。两种 GPU 服务引擎完成；NPU 原生 vLLM/SGLang 初始化失败，明确保留数值覆盖缺口。未设精度验收阈值。

### DeepSeek-V4.1-Flash-1017

[精度报告](.github/deepseek41-precision-report.md)和[已核验证据](site/evidence/deepseek41-precision-20261106.json)比较固定 revision 的 GPU 原版 TP4 实现与明确适配为 BF16 的 NPU TP8/EP8/offload48 配置。覆盖 191/275 项配对张量观测和全部 3,168 个目标评分；NPU 实际 32K 完整前向及精确分段请求已完成。训练、部分算子范围及被拒绝的原生链路仍保留数值缺口。权重转换核验不证明后端算术等价；未设精度验收阈值。

独立的[公开结果指标](site/data/deepseek41-adapted-core-20261109.json)新增 84 项缓存/掩码打包观测、实际数组全局归约，以及输出头 LoRA 所用的 380 次新鲜冻结骨干前向。[公开结果指标](site/data/deepseek41-full40-attention-20261109.json)使用实际原生操作数，对 rank 0 的 480 个层前向、1,920 行已保存 query 做独立回放；两者均可在[写作页](site/paper.html)读取。原始 191/275 库存保持冻结；FP64 注意力回放不模拟原生舍入。新增[公开结果指标](site/data/deepseek41-paired-output-head95-20261109.json)，包含每侧 95 次同坐标调用（83 次更新、12 次保留评估）、独立公式回放及各自续跑/重复轨迹逐位一致性。跨运行时初始 A 近似但不逐位相等，后续优化器状态为独立轨迹。SGLang 与 vLLM 服务数值复核已完成，vLLM 的 Core77 生命周期证据缺口仍保留。这些补充不代表全模型反向或精度验收通过。

- **DeepSeek-V4.1-Flash-1017 — 已复核服务证据（2026 年 11 月 9 日 UTC）。**
  [英文报告](.github/deepseek41-services-final-20261109/report.md)覆盖跨五个 Engine 生命周期的全部 277 项 SGLang 任务，以及八张 H200 上同一 Engine 的 216 项 vLLM TP8/EP8 任务。原主机原始数组数学与 portable 独立复核已完成，vLLM 的 820 行正式完整词表数组已在本地回放。原 GPU **191/275**、原 NPU **269/275** 与另行适配的 paired **275/275** 单列，保留全部 3,168 个目标坐标；95 次输出头研究保持主干冻结。
  Core77 缺少最终恢复回执，原严格生命周期审查仍为失败；八个数据 worker 的终结回执及精确 namespace 清理已核验。12 项科学缺口继续保留。完整词表归一化、相同因果历史配对与空验收阈值保持不变，GPU 不作为数学 oracle。[服务验证报告](.github/deepseek41-services-final-20261109/report.md)提供当前报告、科学图及精确 JSON/CSV 定位。历史 ZIP、原始张量和完整执行日志仅在本地保留。不作发布或完整主干训练完成声明。

[HTTP128 指标](site/data/deepseek41-http128-20261109.json) · [公开结果指标](site/data/deepseek41-http128-20261109.json)。

### 精度修复

在固定的历史软件栈上，修复 Qwen3.5-9B-Base 的零中心 RMSNorm 权重语义。验证覆盖 NPU 原生算子的前向/反向、权重保存与加载，以及加载权重字节不变时的原始 TP4 模型前向复验。

另用小型原生 CPU 工作负载验证了以下历史缺陷的回补：

| 库 / 固定版本 | 修复内容 | 验证范围 |
|---|---|---|
| TRL 0.7.10 | IPO 的 completion logprob 平均归约 | 原生 DPOTrainer loss、全部梯度和一次 SGD 更新；6 项回归 |
| PEFT 0.17.1 | SVD adapter 合并重复应用 LoRA 缩放 | 前向、输入梯度、checkpoint 加载及合并；24 组工作负载 |
| Transformers 4.51.3 | Qwen2 attention mask 忽略关闭滑动窗口的配置 | Eager/SDPA 完整前向及动态/静态 cache；18 组工作负载和 16 组 mask 检查 |

[修复历史](https://satsuki2486441738.github.io/PrecisionRSI/)记录受影响版本、上游修复版本、补丁、前后指标和验证范围。


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

[比较看板](https://satsuki2486441738.github.io/PrecisionRSI/)读取已发布的评测与修复记录。[历史归档](https://satsuki2486441738.github.io/PrecisionRSI/archive.html)保留推理 / SFT / RL 的配置、数值指标、覆盖率和证据，以及按库分组的修复和验证记录。记录及公开证据维护在 [site/](site/README.md)。

[RSI 迭代看板](https://satsuki2486441738.github.io/PrecisionRSI/rsi.html)展示当前运行、候选变更、独立评测结果和分阶段时间/token 开销，单独记录基线复用。


[结果指标页](site/paper.html)提供公开报告、科学图、[JSON](site/data/paper-results.json)和[CSV 数值索引](site/data/paper-results.csv)。公开内容保留来源哈希、实际配置、覆盖范围、失败与未验证项，数值字段不代表独立样本。历史 ZIP、原始张量和完整执行日志仅在本地保留，不提供在线下载。[本地证据索引](site/data/local-evidence.json)记录归档名称、大小和 SHA256。哈希完整性不表示数值验收通过。
