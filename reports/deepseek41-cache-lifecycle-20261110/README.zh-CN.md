# DeepSeek 物理缓存与 Core 生命周期增量索引

日期：2026-11-10。GPU 本轮限定补充已完成独立复核；NPU 物理状态补测等待 SSH 连接恢复。整项任务保持未完成，定时任务保持用户要求的暂停。

- `report.md`：英文研究报告，区分采集、数学复核、生命周期与 namespace 清理。
- [canonical metrics](../../site/data/deepseek41-cache-lifecycle-20261110.json)：精确测量值、统计口径、12项既有科学缺口和源证据SHA。
- `physical-cache-bounded-diagnostics.svg`：标准科学图。
- C1：512快照、1408原始字节数组、2104科学文件；真实非零缓存/logit差异保留。原采集任务的gate超时失败不改写。
- C2：32K历史、八个选定层、末四位置；同一live链复核与独立本地原数组回放完成，C1/C2是不同Engine。
- Core：默认5秒等待造成强制结束；案例cleanup改为原生timeout60，实际9.486秒原生退出返回、严格生命周期和精确namespace清理通过。旧Core77与Core117记录不追溯修改。
- parent/Core映射关闭；八个data worker的live alias保留记录按原样保存，另以native退出和namespace销毁说明后续清理。

原GPU191/275、原NPU269/275与另行adapted paired275/275单列。输出头95不是主干训练；没有数值验收阈值、物理缓存等价或硬件独因结论。旧报告和旧ZIP原字节保留。
