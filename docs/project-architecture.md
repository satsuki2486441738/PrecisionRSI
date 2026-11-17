# NumericBridge：项目架构

**定位：面向 RL 的精度对比诊断与修复优化框架。** 依托可复用 skills、研究运行器与社区记录，围绕硬件 × 模型 × RL 框架 × 训推后端 × 适配库版本 × 配置，交付经过原任务验证的组合、配置或补丁；社区提供已知问题和候选方案的检索及结果沉淀。

本轮在合并后的 main 上完成第一轮模块边界整理，保留当前 NumericBridge 前端与全部历史测量。现有运行接口和可复用方案见各模块文档。

## 两条业务主线

- **精度比较**：在明确的输入、初态、历史和覆盖范围内测量、解释差异。
- **问题修复**：建立正确性依据，定位因果，修改实现，在原始负载上验证修复与代价。

RL 重点包括推理、参数更新和连续长计算图，不能只检查 rollout 输出。比较差异不自动等于缺陷，GPU 不是数学 oracle，完成不代表达到数值验收。组件通过不等于完整 RL 通过，未定义验收阈值保持空值。

## 代码分层

```text
Skills / 可选 precision_rsi → precision_core
社区 site/                 → precision_community（记录、校验与显式关系）
CI / 旧 .github 命令        → precision_maintenance → 社区校验与维护工具
```

| 模块 | 当前职责 |
|---|---|
| [framework-adapt](../skills/framework-adapt/SKILL.md) | 原生部署、模型和后端适配 |
| [precision-compare](../skills/precision-compare/SKILL.md) | 推理、SFT、RL 采集、数学比较及比较专用回放 |
| [precision-repair](../skills/precision-repair/SKILL.md) | 正确性依据、诊断、补丁与原任务复验 |
| [precision_core](../precision_core/README.md) | workflow、record_run、资源目录、运行时身份、通用证据传输与完整性工具 |
| [precision_community](../precision_community/README.md) | 比较、修复、测量视图的校验与只读索引，显式关联层 |
| [precision_rsi](../precision_rsi/README.md) | 可选的 Agent 研究、预算、候选独立评估与版本激活；正常导入公共核心 |
| [precision_maintenance](../precision_maintenance/README.md) | 采集、整合规划、统计与公开导出、安装和发布保护；按需加载 |
| site/ | 版本化 JSON 与静态 GitHub Pages，尚无上传 REST 服务或数据库 |
| 本地私有 projects/ | 案例配置、原生实现、输入、原始数组和执行证据 |

三个 installable skills 已实现，operator-optimize 仍是占位。旧 Compare scripts 保留独立 skill 的兼容入口；维护命令统一使用 `precision_maintenance` 模块，不再提供 `.github/*.py` 转发脚本。独立安装使用导出器，将公共包一并放入 Compare。

## 四个边界的落地与范围

1. **公共执行与比较业务**：通用执行和证据迁入 `precision_core`；数学比较与框架采集留在 Compare。RSI 不再修改 `sys.path` 导入 Compare。工作流的工具身份改变会使旧缓存失效，历史运行保留，不自动改成新工具下的已验证结果。
2. **记录与关联**：`Catalog` 为原始记录提供类型化 ID、来源哈希及 JSON Pointer。硬件、库与版本筛选匹配同一侧。`relations.json` 显式连接比较、调查、问题、候选、验证和原负载，检查引用、类型、范围和证据哈希；不推断或回填历史因果链。现有关系集为空。
3. **研究与社区**：社区校验和索引不导入 RSI 或模型运行时，研究迭代可独立启用。社区检索帮助选择候选，实际组合的验证仍由框架完成；没有自动化的社区—RSI 闭环。
4. **维护与 CI**：可复用逻辑迁入 `precision_maintenance`；CI 调用模块命令。`.github/` 只保留 GitHub 平台配置，现行指南位于 `docs/`，最终报告与科学图位于 `reports/`，维护依赖为根目录 `requirements-dev.txt`。

关系校验仅证明结构、引用和字节绑定有效，不能替代人工对因果与数值结论的审查。社区没有记录不代表配置受支持，候选关系不自动升级为已验证修复。

## 数据流与后续工作

用户需求 → 检索与复用 → 原生任务 → 比较 → 必要时调查和修复 → 原任务复验 → 审查后的公开记录。

原始 ID、日期、指标、配置及证据保持不可变；新关系也受追加保护。公开范围与完整私有执行分别标注，原始轨迹、大数组和历史 ZIP 本地保留。统计回放、数组回放、组件和完整原生任务保持各自证据层级。

后续采用 [RL 精度诊断与修复方案](rl-precision-workflow-proposal.md)，选一个能观察实际参数更新的具体组合，先固定轨迹比较，再诊断、修复和原任务复验。使用首例填充真实关联、检验接口；本轮整理没有运行新的设备实验，也未实现社区上传服务。
