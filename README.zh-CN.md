# precision-skill

[English](README.md) | 简体中文

面向昇腾 NPU 与 NVIDIA GPU 执行栈数值诊断的 [Agent Skill](https://agentskills.io/home)。组织推理、SFT 和 RL 的共同输入、环境证据、采集、离线比较与按需定位。

## 安装与使用

可安装的 skill 位于 [skills/precision-skill](skills/precision-skill/)。将整个目录复制到客户端的 skill 位置，保留目录名 `precision-skill`。包结构遵循 [Agent Skills 规范](https://agentskills.io/specification)。

可以这样开始任务：

> 使用相同模型和初态，比较 GPU 与 NPU 的一个 SFT 步骤。复用 GPU 交接包，对齐共享依赖，比较 loss、抽样梯度和实际参数更新。

提供双侧执行栈、硬件和环境访问条件、已有产物及要回答的问题。agent 查明已有事实，仅对当前阶段需要决策的信息集中确认。任一侧都可以先执行，另一侧通过离线交接复用结果。

## 已实现范围

随包提供的 [Python 比较器](skills/precision-skill/scripts/compare_scores.py) 只依赖标准库。schema v2 / core 2.0.0 核验已保存的 full-forward token 评分，计算 token 加权 NLL、ΔNLL、MAE 和全局线性 P99；当前阈值模式为 `thresholds: null`。

工作流同时定义 SFT 的 loss、梯度、更新语义和 RL 固定轨迹依赖。这些任务使用案例专用采集器与校验器；随包评分核心尚未实现训练或 cache 比较。报告分别呈现执行、可比性、数值结果和实际覆盖。

共享实现库使用相同上游发行版本，包括补丁号；平台构建标签和扩展分别记录。异版本比较需明确批准版本配对和范围。已有运行时清单，或能关联到运行的锁定安装与启动记录，均可用于确认环境；缺少专用版本清单文件不直接意味着需要重跑。

## 目录

```text
README.md / README.zh-CN.md        英文与中文项目说明
AGENTS.md / CLAUDE.md              仓库维护规则
skills/precision-skill/
  SKILL.md                        发现元数据与任务入口
  references/                     按主题组织的英文技术指导
  scripts/compare_scores.py       离线评分比较器
tests/                            维护者回归测试
```

私有项目分别维护环境、契约、采集器和运行产物。模型权重、源数据和环境实体保留在共享资源位置，复用准备记录并核对实际加载绑定。

## 文档导航

| 文档 | 用途 |
|---|---|
| [Skill 入口](skills/precision-skill/SKILL.md) | 任务选择与执行入口。 |
| [工作流](skills/precision-skill/references/workflow.md) | 范围、预算、采集及任务交接。 |
| [环境](skills/precision-skill/references/environment.md) | 共享版本、锁文件、运行关联与资源复用。 |
| [推理](skills/precision-skill/references/inference.md) | 固定历史评分与 cache 语义。 |
| [训练](skills/precision-skill/references/training.md) | SFT 单步、RL 依赖与状态分叉。 |
| [比较](skills/precision-skill/references/comparison.md) | Schema、证据、公式与报告状态。 |
| [架构](skills/precision-skill/references/architecture.md) | 包结构、职责和工作目录归属。 |

在仓库根目录运行维护测试：

```bash
python -B -m unittest discover -s tests -v
```

两版 README 描述相同能力；skill 指令、技术参考和维护文档使用英文维护。
