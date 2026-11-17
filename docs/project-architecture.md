# NumericBridge：项目组成

**定位：面向 RL 的精度对比诊断与修复优化框架。主要交付是 Skill + Tools。**

## 使用与交付

用户给出硬件 × 模型 × 训推框架 × 适配库版本 × 配置和期望结果。
Agent 按缺失能力读取 Skill、调用工具，完成比较诊断、必要的修复与原任务复验。
交付经过限定范围验证的配置、版本组合或补丁，以及指标、覆盖和复验证据。
社区提供已知问题及候选方案检索，检索结果仍需在实际环境验证。

## 目录归属

| 目录 | 放什么 |
|---|---|
| [skills/](../skills/precision-compare/SKILL.md) | Agent 工作方法、任务规范与该 Skill 专用工具 |
| [tools/](../tools/README.md) | 实际共用的命令执行、依赖工作流、资源核验和证据工具 |
| [community/](../community/README.md) | site/ 管理网页、公开数据与证据；records/ 管理校验与检索 |
| [scripts/](../scripts/README.md) | 安装、数据导出、仓库与发布校验等维护脚本 |
| 本地 research/（Git 忽略） | 可选的 RSI 研究实验，代码与状态保留本地，不上传 GitHub |
| 本地 tests/（Git 忽略） | 开发回归，仅供本地维护，不上传 GitHub |
| docs/、reports/ | 现行使用指南、最终测评报告和科学图 |
| .github/ | GitHub 自动化配置 |
| 本地私有 projects/、runs/ | 原生案例、资源配置、原始数组、完整日志与过程记录 |

这些目录按用途分工，不是用户必须理解或依次启动的架构层。
三个已实现 Skill 是 framework-adapt、precision-compare、precision-repair；
operator-optimize 仍是占位，不计为已实现能力。

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
社区关联仅记录有证据支持的关系，目前关系集为空，不推断历史因果。

实际案例执行见 [RL 精度诊断与修复方案](rl-precision-workflow-proposal.md)。
