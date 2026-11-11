/* Read-only community projections. Historical IDs, measurements and evidence stay in the source archives. */
(function (global) {
  'use strict';
  const tasks = { inference: '推理', sft: 'SFT / 微调', rl: 'RL / 强化学习', other: '其他 / 未标注' };
  const unique = (xs) => [...new Set(xs.filter(Boolean))];
  function hardware(raw) {
    const labels = global.PrecisionHistory.hardwareLabels(raw);
    const devices = labels.filter((s) => /^(?:CPU$|Ascend\s|NVIDIA\s|AMD Instinct\s)/.test(s));
    return devices.length ? devices : [raw || '未记录'];
  }
  const aliases = { torch: 'PyTorch', 'pytorch (cuda)': 'PyTorch', 'torch-npu': 'torch_npu', torchnpu: 'torch_npu',
    'megatron-core': 'Megatron Core', 'megatron-bridge': 'Megatron Bridge', 'transformer-engine': 'Transformer Engine' };
  const names = ['PyTorch', 'Transformers', 'TRL', 'PEFT', 'Accelerate', 'AReaL', 'Axolotl', 'LMDeploy', 'OpenRLHF', 'ROLL', 'Ray', 'NumPy', 'Python', 'Datasets', 'Triton', 'vLLM'];
  function libraryKey(name) { return aliases[String(name).toLowerCase()] || names.find((n) => n.toLowerCase() === String(name).toLowerCase()) || name; }
  const indexedLibraries = (c) => c.indexLibraries || c.libraries.map((l) => libraryKey(l.name));
  function build(repairData, evaluationData) {
    const rows = [];
    for (const group of global.PrecisionHistory.buildCatalog(repairData)) for (const fix of group.fixes) {
      const view = global.PrecisionRepairPresentation.overview(fix);
      // Keep each environment attached to its own hardware and versions; never form a cross-side product.
      const configs = fix.records.map((r) => ({ hardware: hardware(r.environment?.hardware), rawHardware: r.environment?.hardware || '未记录',
        libraries: r.environment?.libraries || [], model: r.model?.name || '未记录', settings: r.problem?.trigger || '', recordId: r.id, indexLibraries: [libraryKey(r.affected_library || fix.root.affected_library || group.title)] }));
      rows.push({ id: fix.root.id, kind: 'repair', title: view.title, summary: fix.root.problem?.summary || '',
        date: fix.records.map((r) => r.date || '').sort().at(-1), tasks: view.tasks.length ? view.tasks : ['other'], configs,
        affected: fix.root.affected_library || group.title, records: fix.records, source: fix.root,
        url: `./history.html#fix-${encodeURIComponent(fix.root.id)}` });
    }
    for (const r of global.PrecisionHistory.publicEvaluations(evaluationData.evaluations)) {
      rows.push({ id: r.id, kind: 'evaluation', title: r.title, summary: r.summary || '', date: r.date,
        tasks: [tasks[r.task] ? r.task : 'other'], source: r,
        configs: r.sides.map((s) => ({ hardware: hardware(s.hardware), rawHardware: s.hardware || '未记录', libraries: s.libraries || [],
          model: r.model?.name || '未记录', settings: s.settings || {}, recordId: s.id })),
        url: `./history.html#evaluation-${encodeURIComponent(r.id)}` });
    }
    return rows;
  }
  function matchConfigs(row, f) {
    return row.configs.filter((c) => (!f.hardware || c.hardware.includes(f.hardware)) &&
      (!f.library || indexedLibraries(c).includes(f.library)) && (!f.model || c.model === f.model));
  }
  function filter(rows, f = {}) {
    const q = (f.query || '').trim().toLocaleLowerCase();
    return rows.filter((r) => (!f.kind || r.kind === f.kind) && (!f.task || r.tasks.includes(f.task)) && matchConfigs(r, f).length &&
      (!q || JSON.stringify(r).toLocaleLowerCase().includes(q)));
  }
  const api = { build, filter, matchConfigs, tasks, hardware, libraryKey, indexedLibraries };
  if (typeof module !== 'undefined' && module.exports) { module.exports = api; return; }
  global.PrecisionCommunity = api;
  if (!global.document) return;
  // Preserve inbound historical permalinks when the new portal is the landing page.
  if (/^#(?:repairs|evaluations|(?:repair|fix|library|evaluation)-)/.test(location.hash)) {
    location.replace('./history.html' + location.hash); return;
  }
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const fields = ['query', 'hardware', 'task', 'library', 'model', 'kind'];
  let rows = [], limit = 20;
  const values = () => Object.fromEntries(fields.map((id) => [id, $(id).value]));
  function options(id, xs) {
    for (const x of unique(xs).sort((a, b) => a.localeCompare(b))) { const o = el('option', '', x); o.value = x; $(id).append(o); }
  }
  function renderMatrix(filtered, f) {
    const columns = f.task ? [f.task] : Object.keys(tasks);
    const libs = unique(filtered.flatMap((r) => matchConfigs(r, f).flatMap(indexedLibraries))).sort();
    const table = el('table', 'matrix'); table.append(el('caption', 'sr-only', '软件组件与训推任务的记录索引'));
    const head = el('thead'), hr = el('tr'); hr.append(el('th', '', '软件组件 / 观测环境'));
    columns.forEach((t) => { const h = el('th', '', tasks[t]); h.scope = 'col'; hr.append(h); }); head.append(hr); table.append(head);
    const body = el('tbody');
    for (const lib of libs) {
      const tr = el('tr'), h = el('th'); h.scope = 'row'; h.append(el('strong', '', lib), el('span', 'lib-sub', f.hardware || '按记录中的实际硬件筛选')); tr.append(h);
      for (const task of columns) {
        const cell = el('td');
        const matches = filtered.filter((r) => r.tasks.includes(task) && matchConfigs(r, { ...f, library: lib }).length);
        if (!matches.length) cell.append(el('span', 'no-record', '— 暂无记录'));
        else {
          const b = el('button', 'cell'); b.type = 'button';
          const n = matches.filter((r) => r.kind === 'repair').length, e = matches.length - n;
          if (n) b.append(el('span', 'cell-repair', `${n} 项问题 / 修复档案`));
          if (e) b.append(el('span', 'cell-evaluation', `${e} 项评测记录`));
          b.setAttribute('aria-label', `${lib} · ${tasks[task]}：${n} 项问题 / 修复档案，${e} 项评测记录，查看详情`);
          b.addEventListener('click', () => { $('library').value = lib; $('task').value = task; limit = 20; render(); $('results').scrollIntoView({ behavior: 'smooth', block: 'start' }); });
          cell.append(b);
        }
        tr.append(cell);
      }
      body.append(tr);
    }
    table.append(body);
    $('matrix').replaceChildren(libs.length ? table : el('p', 'loading', '当前组合暂无记录。可清除筛选查看其他环境。'));
  }
  function card(row, f) {
    const article = el('details', 'record');
    const summary = el('summary', 'record-summary');
    const badge = el('span', `badge ${row.kind}`, row.kind === 'repair' ? (row.source.status === 'investigating' ? '问题调查中' : '修复档案') : '评测 / 数值观测');
    const title = el('div', 'record-title'); title.append(el('h3', '', row.title), el('p', '', row.summary));
    summary.append(badge, title, el('time', '', row.date), el('span', 'expand', '+')); article.append(summary);
    const content = el('div', 'record-content');
    const configs = matchConfigs(row, f), tags = el('div', 'tags');
    unique(configs.flatMap((c) => c.hardware)).forEach((h) => tags.append(el('span', '', h)));
    row.tasks.forEach((t) => tags.append(el('span', '', tasks[t]))); content.append(tags);
    const table = el('table', 'config-table');
    const head = el('tr'); ['实际硬件', '模型', '库与版本', '触发条件 / Settings'].forEach((s) => head.append(el('th', '', s))); const thead = el('thead'); thead.append(head); table.append(thead);
    const body = el('tbody');
    configs.forEach((c) => { const tr = el('tr'); [c.rawHardware, c.model, c.libraries.map((l) => [l.name, l.version, l.commit].filter(Boolean).join(' ')).join(' · '),
      typeof c.settings === 'string' ? c.settings : JSON.stringify(c.settings, null, 2)].forEach((s) => tr.append(el('td', '', s))); body.append(tr); });
    table.append(body); const wrap = el('div', 'config-wrap'); wrap.append(table); content.append(wrap);
    if (row.kind === 'repair') {
      const r = row.source;
      content.append(el('p', '', `涉及实现：${row.affected}。${r.repair?.summary || ''}`), el('p', 'scope-note', `上游状态：${r.repair?.upstream_status || '未记录'}。验证范围：${r.validation?.scope || '见原记录'}。`));
      const m = r.validation?.metrics?.[0]; if (m) content.append(el('p', 'metric', `${m.label || m.id}：${m.before} → ${m.after} ${m.unit || ''}（该轮修复前 / 后）`));
      content.append(el('p', 'scope-note', `${row.records.length} 轮历史记录；数值验收：${r.validation?.acceptance == null ? '未声明' : JSON.stringify(r.validation.acceptance)}。查看详情核对各轮配置和剩余限制。`));
    } else {
      content.append(el('p', 'scope-note', `执行状态：${row.source.status}；数值验收：${row.source.acceptance == null ? '未声明' : JSON.stringify(row.source.acceptance)}。该记录不自动判为实现缺陷。`));
    }
    const a = el('a', 'detail-link', '查看完整配置、指标与原始证据 ↗'); a.href = row.url; content.append(a); article.append(content); return article;
  }
  function render() {
    const f = values(); let filtered = filter(rows, f);
    filtered.sort((a, b) => ($('sort').value === 'repair' ? Number(b.kind === 'repair') - Number(a.kind === 'repair') : 0) || b.date.localeCompare(a.date));
    $('chips').replaceChildren();
    fields.filter((id) => $(id).value).forEach((id) => { const b = el('button', 'chip', `${id === 'query' ? $('query').value : $(id).selectedOptions[0].textContent} ×`); b.type = 'button'; b.setAttribute('aria-label', `取消筛选 ${b.textContent}`); b.addEventListener('click', () => { $(id).value = ''; limit = 20; render(); }); $('chips').append(b); });
    renderMatrix(filtered, f);
    $('count').textContent = `${filtered.length} / ${rows.length}`;
    $('selection').textContent = f.hardware ? `观测硬件：${f.hardware}。记录仅覆盖原文中的版本、模型和配置。` : '当前展示所有已记录硬件。选择具体设备可缩小到你的使用环境。';
    $('records').replaceChildren(...filtered.slice(0, limit).map((r) => card(r, f)));
    if (!filtered.length) $('records').append(el('p', 'empty-state', '没有匹配记录。暂无记录不代表该配置已通过验证。'));
    $('more').hidden = filtered.length <= limit;
    const params = new URLSearchParams(Object.entries(f).filter(([, v]) => v));
    const url = new URL(location.href); url.search = params.toString(); history.replaceState(null, '', url);
  }
  async function load() {
    try {
      const data = await Promise.all(['repairs', 'evaluations'].map(async (name) => { const r = await fetch(`./data/${name}.json`); if (!r.ok) throw Error('HTTP'); const d = await r.json(); if (d.schema_version !== 1 || !Array.isArray(d[name])) throw Error('schema'); return d; }));
      rows = build(...data);
      options('hardware', rows.flatMap((r) => r.configs.flatMap((c) => c.hardware)));
      options('library', rows.flatMap((r) => r.configs.flatMap(indexedLibraries)));
      options('model', rows.flatMap((r) => r.configs.map((c) => c.model)));
      const stats = [[rows.filter((r) => r.kind === 'repair').length, '问题与修复档案'], [rows.filter((r) => r.kind === 'evaluation').length, '公开评测记录'],
        [unique(rows.flatMap((r) => r.configs.flatMap((c) => c.hardware))).length, '已记录硬件'], [unique(rows.flatMap((r) => r.configs.flatMap(indexedLibraries))).length, '涉及软件组件']];
      $('stats').replaceChildren(...stats.map(([n, label]) => { const d = el('div'); d.append(el('strong', '', n), el('span', '', label)); return d; }));
      const params = new URLSearchParams(location.search); fields.forEach((id) => { if (params.has(id)) $(id).value = params.get(id); }); render();
    } catch { $('matrix').replaceChildren(el('p', 'loading', '公开数据加载失败，请重试或查看历史页面。')); const b = el('button', 'reset', '重新加载'); b.addEventListener('click', () => location.reload()); $('matrix').append(b); }
  }
  $('filters').addEventListener('submit', (e) => e.preventDefault());
  fields.forEach((id) => $(id).addEventListener(id === 'query' ? 'input' : 'change', () => { limit = 20; render(); }));
  $('filters').addEventListener('reset', () => setTimeout(() => { limit = 20; render(); }, 0));
  $('sort').addEventListener('change', render);
  $('more').addEventListener('click', () => { limit += 20; render(); });
  load();
})(globalThis);
