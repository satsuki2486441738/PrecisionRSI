/* Library groups contain independent fixes; each fix retains its verification records. */
(function (global) {
  'use strict';
  function buildCatalog(data) {
    const records = new Map();
    for (const record of data.repairs) {
      if (records.has(record.id)) throw new Error('Duplicate record ID');
      records.set(record.id, record);
    }
    const definitions = new Map();
    for (const group of data.library_groups || []) {
      if (definitions.has(group.id)) throw new Error('Duplicate library group ID');
      definitions.set(group.id, group);
    }
    function rootOf(record) {
      const seen = new Set();
      while (record.related_repair_id) {
        if (seen.has(record.id)) throw new Error('Cyclic repair relationship');
        seen.add(record.id);
        record = records.get(record.related_repair_id);
        if (!record) throw new Error('Missing parent repair');
      }
      return record;
    }
    const fixes = new Map();
    for (const record of records.values()) {
      const root = rootOf(record);
      const definition = definitions.get(root.library_group_id);
      if (!definition) throw new Error('Missing library group');
      if (record.library_group_id && record.library_group_id !== root.library_group_id) throw new Error('Conflicting follow-up library group');
      if (!fixes.has(root.id)) fixes.set(root.id, { root, records: [] });
      fixes.get(root.id).records.push(record);
    }
    const groups = new Map();
    for (const fix of fixes.values()) {
      const id = fix.root.library_group_id;
      if (!groups.has(id)) groups.set(id, { ...definitions.get(id), fixes: [] });
      fix.records.sort((a, b) => String(b.date).localeCompare(String(a.date)));
      groups.get(id).fixes.push(fix);
    }
    return [...groups.values()].sort((a, b) => a.title.localeCompare(b.title));
  }
  function counts(groups) {
    return { groups: groups.length, fixes: groups.reduce((n, g) => n + g.fixes.length, 0),
      verifications: groups.reduce((n, g) => n + g.fixes.reduce((m, f) => m + f.records.length, 0), 0) };
  }
  function filterCatalog(groups, filters) {
    const query = (filters.query || '').trim().toLocaleLowerCase();
    return groups.filter((g) => !filters.library || g.components.includes(filters.library)).map((group) => ({ ...group,
      fixes: group.fixes.map((fix) => ({ ...fix, records: fix.records.filter((record) =>
        (!filters.model || record.model?.name === filters.model) &&
        (!filters.hardware || record.environment?.hardware === filters.hardware) &&
        (!query || JSON.stringify({ title: group.title, fix: fix.root.title, cause: fix.root.cause, record }).toLocaleLowerCase().includes(query)))
      })).filter((fix) => fix.records.length)
    })).filter((group) => group.fixes.length);
  }
  const taskLabels = { inference: '推理', sft: 'SFT', rl: 'RL' };
  function evaluationLibraries(record) {
    return [...new Set(record.sides.flatMap((side) => side.libraries.map((lib) => lib.name)))];
  }
  function filterEvaluations(records, filters = {}) {
    const query = (filters.query || '').trim().toLocaleLowerCase();
    return records.filter((record) =>
      (!filters.task || record.task === filters.task) &&
      (!filters.library || evaluationLibraries(record).includes(filters.library)) &&
      (!filters.model || record.model.name === filters.model) &&
      (!filters.hardware || record.sides.some((side) => side.hardware === filters.hardware)) &&
      (!query || JSON.stringify(record).toLocaleLowerCase().includes(query))
    ).sort((a, b) => b.date.localeCompare(a.date));
  }
  function evaluationGroups(records) {
    return Object.entries(taskLabels).map(([id, title]) => ({ id, title,
      records: records.filter((record) => record.task === id) })).filter((group) => group.records.length);
  }
  function evaluationCounts(records) {
    return { experiments: records.length,
      comparisons: records.reduce((n, record) => n + record.comparisons.length, 0),
      hardware: new Set(records.flatMap((record) => record.sides.map((side) => side.hardware))).size };
  }
  function routeForHash(hash) {
    let id;
    try { id = decodeURIComponent(hash.replace(/^#/, '')); } catch { return null; }
    if (id === 'evaluations' || id.startsWith('evaluation-')) return { kind: 'evaluations', id };
    if (!id || id === 'repairs' || /^(repair|fix|library)-/.test(id)) return { kind: 'repairs', id };
    return null;
  }
  const api = { buildCatalog, counts, filterCatalog, taskLabels, evaluationLibraries,
    filterEvaluations, evaluationGroups, evaluationCounts, routeForHash };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.PrecisionHistory = api;
})(globalThis);
