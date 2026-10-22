/* Editorial repair summaries. Historical measurements stay in repairs.json.
   New records may supply presentation:{title,tasks,version,sharedTraining}.
   Task tags describe the affected workflow, not additional verified workloads. */
(function (global) {
  'use strict';
  const labels = { inference: '推理', sft: 'SFT', rl: 'RL' };
  const known = {
    'mindspeed-rmsnorm-zero-centered-gamma': { title: 'RMSNorm 参数适配不一致', tasks: ['inference', 'rl'] },
    'transformers-4513-qwen2-disabled-window-backport': { title: 'Qwen2 注意力掩码错误', tasks: ['inference'] },
    'transformers-4533-trainer-label-smoothing-backport': { title: 'Trainer 梯度累积归一化错误', tasks: ['sft'] },
    'peft-0171-lora-svd-double-scaling-backport': { title: 'LoRA 适配器组合错误', tasks: ['sft'] },
    'peft-0171-dora-conv1d-unmerge-orientation': { title: 'DoRA 合并后权重恢复错误', tasks: ['sft'] },
    'trl-0710-ipo-completion-average-backport': { title: 'IPO 偏好目标计算错误', tasks: ['rl'] },
    'trl-0710-ppo-response-mask-backport': { title: 'PPO 训练 token 选择错误', tasks: ['rl'] },
    'trl-0710-ppo-whitening-tail-leakage': { title: 'PPO 回报计算被 padding 污染', tasks: ['rl'] },
    'pytorch-230-asgd-foreach-schedule-backport': { title: 'ASGD 学习率更新错误', tasks: ['sft', 'rl'], sharedTraining: true, version: '2.3.0 源码 · 2.6.0 runtime' },
    'pytorch-260-radam-foreach-unrectified-zero-gradient': { title: 'RAdam 参数更新产生 NaN', tasks: ['sft', 'rl'], sharedTraining: true }
  };
  function overview(fix) {
    const record = fix.root;
    const view = record.presentation || known[record.id] || {};
    const tasks = [...new Set((view.tasks || record.tasks || [record.task]).filter((task) => labels[task]))];
    const library = (record.environment?.libraries || []).find((lib) => lib.name === record.affected_library);
    const version = view.version || library?.version || (library?.commit ? `commit ${library.commit.slice(0, 7)}` : '版本未记录');
    return { title: view.title || record.title, tasks, sharedTraining: Boolean(view.sharedTraining),
      library: record.affected_library || record.library_group_id, version,
      metric: (record.validation?.metrics || [])[0] || null };
  }
  function filterTask(groups, task) {
    if (!task) return groups;
    return groups.map((group) => ({ ...group, fixes: group.fixes.filter((fix) => overview(fix).tasks.includes(task)) }))
      .filter((group) => group.fixes.length);
  }
  function compactNumber(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '未记录';
    if (value === 0) return '0';
    const magnitude = Math.abs(value);
    if (magnitude < .001 || magnitude >= 100000) return value.toExponential(2).replace('e+', 'e');
    return String(Number(value.toPrecision(4)));
  }
  const api = { labels, overview, filterTask, compactNumber };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.PrecisionRepairPresentation = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
