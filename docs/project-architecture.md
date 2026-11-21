# NumericBridge：项目组成

**定位：面向 RL 的精度对比诊断与修复优化框架。主要交付是 Skill + Tools。**

## 使用与交付

Agent 从 [NumericBridge Skill](../skills/numericbridge/SKILL.md) 进入。[流程规范](../skills/numericbridge/references/workflow.md)定义执行分支和交接。Agent 按当前步骤读取适配、对比或修复 Skill，沿用已确认的需求、运行身份和证据位置。RL 采集与指标见 [RL 方法](../skills/precision-compare/references/rl.md)。

[RL 首例设计](rl-precision-workflow-proposal.md)用于选择研究案例和设计对照。[整模型评测手册](model-precision-evaluation-plan.md)用于明确要求整模型覆盖的任务。

## 目录归属

| 目录 | 放什么 |
|---|---|
| [skills/](../skills/numericbridge/SKILL.md) | NumericBridge 总入口、各阶段工作方法和专用工具 |
| [tools/](../tools/README.md) | 实际共用的命令执行、依赖工作流、资源核验和证据工具 |
| [community/](../community/README.md) | site/ 管理网页、公开数据与证据；records/ 管理校验与检索 |
| [scripts/](../scripts/README.md) | 安装、数据导出、仓库与发布校验等维护脚本 |
| 本地 research/（Git 忽略） | 可选的 RSI 研究实验，代码与状态保留本地，不上传 GitHub |
| 本地 tests/（Git 忽略） | 开发回归，仅供本地维护，不上传 GitHub |
| docs/、reports/ | 现行使用指南、最终测评报告和科学图 |
| .github/ | GitHub 自动化配置 |
| 本地私有 projects/、runs/ | 原生案例、资源配置、原始数组、完整日志与过程记录 |

这些目录按用途分工，不是用户必须理解或依次启动的架构层。
安装包包含 numericbridge、framework-adapt、precision-compare、precision-repair。仓库中的 operator-optimize 是预留项。

## 工具放置与安装

数组/分布比较、采集适配和比较专用回放继续放在 Compare 的 scripts/ 中。
被多个 Skill 或研究运行器实际复用的工具才放在根目录 tools/。
Skill 脚本保留调用入口，共用工具只维护一份实现。

使用 `python -B -m scripts install --output NEW_DIRECTORY` 导出独立 skills。
导出器把所需 tools/ 一并放入 Compare 的 scripts/，离开仓库也可运行，
不需要 research/、维护脚本或网站服务。维护命令见 scripts/README.md。

## 可选研究与身份

日常 Skill 使用不依赖 RSI。research/ 不随公开仓库分发，仅在本地明确需要自动研究迭代时启用。
研究快照及版本身份包含 research/、skills/ 和实际依赖的 tools/；
工具改变后必须重新验证受影响范围，历史运行及旧版本回执不自动升级。
旧目录改名不修改历史证据中的源路径，也不追溯补造缺失证据。

## 结果边界

比较差异不自动等于缺陷，GPU 不是数学 oracle，采集完成不代表数值验收。
组件结果不等于完整 RL、参数更新或长计算图通过；未声明阈值保持空值。
公开内容遵循[发布规则](publication.md)，原始结果和引用证据保持不可变。
社区以案例关联评测、问题、多个方案和原任务复验；同一问题可对应本地补丁、配置策略或较新版本的修复途径。
已整理全部 130 条评测、20 条修复历史和 7783 条测量视图，见[完整清单](../reports/community-record-inventory-20261117.md)。
其中 26 条 Skills/Agent 评测单列为工作流评测；测量视图不计为独立实验。
[records 的检索接口](../community/records/README.md)支持按实际硬件、模型、库版本和任务筛选案例；网页的精度案例库（`#/cases`）展示问题、修复方案及各方案的复验结果。
只有明确来源支持的关系才入库；源码检查、原任务复验、候选选择与数值验收分别表达。
