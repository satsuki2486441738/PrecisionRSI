# NPU512 物理缓存写作索引

此前因连接受阻而未执行的 NPU 补测现已完成：同一 TP8/EP8 Engine 上四次512-token请求，8个rank原始数组、独立隔离回放、原主机数学复核及精确进程清理均通过。整体科学覆盖仍未完成，验收阈值未定义，物理缓存等价性未证。

- `report.md`：英文方法、实测结果及范围。
- `summary.json`：精确指标、相关观测坐标和不可变证据SHA。
- `npu512-physical-cache-diagnostics.svg`、`.pdf`、`.png`：rank0的三个源层pool/norm诊断；横轴为分类位置，TP观测相关。
- `README.md`：对应英文索引。

2304个状态环行写入及96项compressor/indexer原始别名已核验。全段与253+259分段末四位置logits存在差异，RMSE为0.4059744611614696；各配置内部控制/采集组head字节相同。原始GPU191/275、NPU269/275及另行adaptedpaired275/275分开保留。原12项科学缺口与早期连接失败快照均不改写；不宣称32K/all40、projection/RoPE、原生ownscore、backbone训练或硬件独因。定时任务保持用户暂停，未发布。
