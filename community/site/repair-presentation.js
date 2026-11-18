/* Editorial repair summaries. Historical measurements stay in repairs.json.
   New records may supply presentation:{title,tasks,version,sharedTraining}.
   Task tags describe the affected workflow, not additional verified workloads. */
(function (global) {
  'use strict';
  const labels = { inference: '推理', sft: 'SFT', rl: 'RL' };
  let content = {};
  function setContent(records) { content = records; }
  function overview(fix) {
    const record = fix.root;
    const view = record.presentation || {}, description=record.content||content['repair:'+record.id]||{};
    const tasks = [...new Set((description.workflows || view.tasks || record.tasks || [record.task]).filter((task) => labels[task]))];
    const library = (record.environment?.libraries || []).find((lib) => lib.name === record.affected_library);
    const version = library?.version || (library?.commit ? '源码版本' : view.version || '版本未记录');
    return { title: description.title?.zh || description.title?.en || view.title || record.title, tasks,
      sharedTraining: Boolean(view.sharedTraining || tasks.includes('sft')&&tasks.includes('rl')),
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
  const api = { labels, overview, filterTask, compactNumber, setContent };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.PrecisionRepairPresentation = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
