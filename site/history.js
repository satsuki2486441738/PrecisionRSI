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
  function evaluationModel(record) {
    const name = record.model.name.trim().replace(/^(?:Qwen|HuggingFaceTB)\//i, '');
    // Fixture weights, seeds and derived configurations remain in the raw record.
    const fixture = name.match(/\b(Qwen\d+(?:\.\d+)?|Llama|Megatron\s+GPT)\b/i);
    if (fixture && /\btiny\b|随机|\brandom\b/i.test(name) && !/\d+(?:\.\d+)?[BM]\b/i.test(name)) {
      const family = fixture[1].replace(/^qwen/i, 'Qwen').replace(/^llama$/i, 'Llama').replace(/^megatron\s+gpt$/i, 'Megatron GPT');
      return `tiny ${family}`;
    }
    if (/^SmolLM2[（(]134,?515,?008\s*参数快照[）)]$/i.test(name)) return 'SmolLM2-135M-Instruct';
    return name.replace(/\s*[·（(]\s*(?:untied\b|seed(?:ed)?(?=\b|\d|[-_:])|种子|snapshot\b|快照)[\s\S]*$/i, '').trim();
  }
  function hardwareLabels(raw) {
    const labels = new Set();
    // Split platform lists after removing descriptive parentheses. Parentheses
    // naming another platform are retained so every peer platform is counted.
    const text = String(raw || '').replace(/[（(]([^）)]*)[）)]/g, (whole, detail) =>
      /\b(?:CPU\b|Ascend(?=\s|\d)|NVIDIA\b|GeForce\b|RTX(?=\s|[a-z]?\d)|AMD\b|Instinct\b|MI\d{2,3}[a-z]*(?![a-z\d])|[AHVBL]\d{2,3}S?(?![a-z\d]))/i.test(detail) ? detail : ' ');
    for (const part of text.split(/\s*(?:[+/,;；、|]|\band\b)\s*/i)) {
      const found = [];
      function collect(pattern, label) {
        for (const match of part.matchAll(pattern)) found.push(label(match));
      }
      collect(/\bCPU\b/gi, () => 'CPU');
      collect(/\bAscend\s*(?:NPU\s*)?(\d{3}[a-z]?)(?![a-z\d])/gi, (match) => `Ascend ${match[1].toUpperCase()}`);
      collect(/\bAscend\s+NPU(?!\s*\d{3})/gi, () => 'Ascend NPU');
      collect(/\b(?:NVIDIA\s+)?(?:GeForce\s+)?RTX\s*([a-z]?\d{3,4})(?:\s*(Ti|SUPER|Ada(?:\s+Generation)?))?(?![a-z\d])/gi, (match) => {
        const model = match[1].toUpperCase();
        const suffix = match[2] ? ` ${/^ti$/i.test(match[2]) ? 'Ti' : /^super$/i.test(match[2]) ? 'SUPER' : 'Ada'}` : '';
        const family = /GeForce/i.test(match[0]) || (/^[2-5]0[5-9]0$/.test(model) && !/Ada/i.test(suffix)) ? 'GeForce ' : '';
        return `NVIDIA ${family}RTX ${model}${suffix}`;
      });
      collect(/\b(?:NVIDIA\s+)?([AHVBL]\d{2,3}S?)(?![a-z\d])/gi, (match) => `NVIDIA ${match[1].toUpperCase()}`);
      collect(/\b(?:AMD\s+)?(?:Instinct\s+)?(MI\d{2,3}[a-z]*)(?![a-z\d])/gi, (match) => `AMD Instinct ${match[1].toUpperCase()}`);
      if (found.length) {
        found.forEach((label) => labels.add(label));
      } else {
        const fallback = part.replace(/^\s*\d+\s*[×x*]\s*/i, '').split(/\s*·\s*/)[0].trim();
        if (fallback && !/^(?:(?:npu|cuda|gpu|device)\s*:?\s*\d+|physical\s*\d+|\d+(?:\s*(?:cards?|GPUs?|NPUs?|卡))?|\d+\s*(?:Gi?B|Mi?B)(?:\s+HBM\d*)?|(?:x86_64|aarch64|WSL)(?:\s.*)?)$/i.test(fallback)) labels.add(fallback);
      }
    }
    return [...labels];
  }
  function evaluationHardware(record) {
    return [...new Set(record.sides.flatMap((side) => hardwareLabels(side.hardware)))];
  }
  function evaluationFacets(records) {
    const sorted = (values) => [...new Set(values)].sort((a, b) => a.localeCompare(b, 'en'));
    return { models: sorted(records.map(evaluationModel)), hardware: sorted(records.flatMap(evaluationHardware)) };
  }
  function filterEvaluations(records, filters = {}) {
    const query = (filters.query || '').trim().toLocaleLowerCase();
    return records.filter((record) =>
      (!filters.task || record.task === filters.task) &&
      (!filters.library || evaluationLibraries(record).includes(filters.library)) &&
      (!filters.model || evaluationModel(record) === filters.model) &&
      (!filters.hardware || evaluationHardware(record).includes(filters.hardware)) &&
      (!query || JSON.stringify({ record, model: evaluationModel(record), hardware: evaluationHardware(record) }).toLocaleLowerCase().includes(query))
    ).sort((a, b) => b.date.localeCompare(a.date));
  }
  function evaluationGroups(records) {
    return Object.entries(taskLabels).map(([id, title]) => ({ id, title,
      records: records.filter((record) => record.task === id) })).filter((group) => group.records.length);
  }
  function evaluationCounts(records) {
    return { experiments: records.length,
      comparisons: records.reduce((n, record) => n + record.comparisons.length, 0),
      hardware: new Set(records.flatMap(evaluationHardware)).size };
  }
  function routeForHash(hash) {
    let id;
    try { id = decodeURIComponent(hash.replace(/^#/, '')); } catch { return null; }
    if (id === 'evaluations' || id.startsWith('evaluation-')) return { kind: 'evaluations', id };
    if (!id || id === 'repairs' || /^(repair|fix|library)-/.test(id)) return { kind: 'repairs', id };
    return null;
  }
  const api = { buildCatalog, counts, filterCatalog, taskLabels, evaluationLibraries,
    evaluationModel, hardwareLabels, evaluationHardware, evaluationFacets,
    filterEvaluations, evaluationGroups, evaluationCounts, routeForHash };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.PrecisionHistory = api;
})(globalThis);
