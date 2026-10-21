'use strict';

const $ = (id) => document.getElementById(id);
const controls = ['search', 'task', 'library', 'model', 'hardware'];
let archive = null;
let repairs = [];
let catalog = [];
let evaluations = [];
let activeKind = 'repairs';
let requestSerial = 0;
let pendingRequest = null;
const historyAPI = globalThis.PrecisionHistory;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}
const list = (value) => Array.isArray(value) ? value : [];
const text = (value, fallback = '未记录') => value === null || value === undefined || value === '' ? fallback : String(value);
const libraries = (record) => list(record.environment?.libraries);
const libraryLabel = (lib) => [lib.name, lib.version, lib.commit ? `commit ${lib.commit}` : ''].filter(Boolean).join(' · ');

// Permit bundled evidence and public HTTPS URLs; reject local paths and private hosts.
function publicURL(value) {
  if (typeof value === 'string' && /^(?:\.\/)?evidence\/[a-zA-Z0-9_./-]+$/.test(value) && !value.split('/').includes('..')) return value;
  try {
    const url = new URL(value);
    const host = url.hostname.toLowerCase();
    if (url.protocol !== 'https:' || url.username || url.password || url.port) return null;
    if (!host.includes('.') || /[:\[\]]/.test(host) || /^\d+(\.\d+){3}$/.test(host)) return null;
    if (/(^|\.)(localhost|local|internal|lan|home|test|invalid|example)$/.test(host)) return null;
    return url.href;
  } catch { return null; }
}
function paragraph(parent, label, value) {
  if (value === undefined || value === null || value === '') return;
  const p = node('p');
  p.append(node('strong', '', `${label}：`), document.createTextNode(text(value)));
  parent.append(p);
}
function section(title) {
  const element = node('section', 'detail-section');
  element.append(node('h6', '', title));
  return element;
}
function bulletList(parent, values) {
  if (!list(values).length) return;
  const ul = node('ul');
  values.forEach((value) => ul.append(node('li', '', value)));
  parent.append(ul);
}
function formatNumber(value) {
  // String preserves the supplied JSON number without rounding tiny error metrics to zero.
  return typeof value === 'number' && Number.isFinite(value) ? String(value) : '未记录';
}
function metricGraphic(metric) {
  const item = node('div', 'metric');
  const name = node('p', 'metric-name', text(metric.label, text(metric.id)));
  name.append(node('span', '', metric.unit ? `(${metric.unit})` : '（无量纲）'));
  item.append(name);
  const values = [metric.before, metric.after];
  const drawable = values.every((value) => typeof value === 'number' && Number.isFinite(value) && value >= 0);
  const maximum = drawable ? Math.max(...values) : 0;
  values.forEach((value, index) => {
    const row = node('div', 'bar-row');
    row.append(node('span', '', index ? '修复后' : '修复前'));
    const track = node('div', 'track');
    track.setAttribute('aria-hidden', 'true');
    if (drawable) {
      const bar = node('div', `bar${index ? ' after' : ''}`);
      bar.style.width = `${maximum ? value / maximum * 100 : 0}%`;
      track.append(bar);
    } else track.hidden = true;
    row.append(track, node('span', 'metric-value', `${formatNumber(value)}${metric.unit ? ` ${metric.unit}` : ''}`));
    item.append(row);
  });
  return item;
}
function renderDiagnosis(record) {
  const diagnosis = node('div', 'diagnosis');
  [['根因 / CAUSE', record.cause?.summary], ['修复 / REPAIR', record.repair?.summary]].forEach(([label, summary]) => {
    const block = node('div');
    block.append(node('span', 'small-label', label), node('p', '', text(summary)));
    diagnosis.append(block);
  });
  return diagnosis;
}
function openAncestors(element) {
  for (let parent = element.parentElement; parent; parent = parent.parentElement) {
    if (parent.tagName === 'DETAILS') parent.open = true;
  }
}
function renderCard(record) {
  const card = node('article', 'repair-card verification-card');
  card.id = `repair-${record.id}`;
  const header = node('div', 'card-header');
  const rail = node('div', 'date-rail');
  const date = node('time', '', text(record.date));
  if (/^\d{4}-\d{2}-\d{2}$/.test(record.date)) date.dateTime = record.date;
  rail.append(date);
  const heading = node('div', 'card-heading');
  const top = node('div', 'card-topline');
  top.append(node('span', `pill${record.status === 'validated' ? ' validated' : ''}`, record.status === 'validated' ? '✓ 验证完成' : text(record.status, '状态未记录')));
  const acceptance = record.validation?.acceptance;
  top.append(node('span', `pill${acceptance == null ? ' pending' : ''}`, acceptance == null ? '验收 · 待用户判定' : `验收 · ${typeof acceptance === 'object' ? text(acceptance.status) : text(acceptance)}`));
  const link = node('a', 'permalink', '条目链接 ↗');
  link.href = `#${encodeURIComponent(card.id)}`;
  link.setAttribute('aria-label', `链接至：${text(record.title)}`);
  link.addEventListener('click', () => { openAncestors(card); card.querySelector('details').open = true; });
  const mobileDate = node('time', 'mobile-date', text(record.date));
  if (/^\d{4}-\d{2}-\d{2}$/.test(record.date)) mobileDate.dateTime = record.date;
  top.append(link, mobileDate);
  const title = node('h5', '', text(record.verification_title, text(record.title)));
  heading.append(top, title, node('p', 'card-summary', text(record.problem?.summary)));
  const tags = node('div', 'tags');
  [record.model?.name, record.environment?.hardware].filter(Boolean).forEach((tag) => tags.append(node('span', '', tag)));
  heading.append(tags);
  header.append(rail, heading);
  const body = node('div', 'card-body');

  const metrics = list(record.validation?.metrics);
  const panel = node('section', 'metric-panel');
  const panelHeading = node('div', 'metric-heading');
  panelHeading.append(node('h4', '', '验证指标'));
  const legend = node('div', 'legend');
  legend.setAttribute('aria-hidden', 'true');
  legend.append(node('span', '', '修复前'), node('span', '', '修复后'));
  if (metrics.length) panelHeading.append(legend);
  panel.append(panelHeading);
  if (metrics.length) {
    const graphics = node('div', 'metrics');
    metrics.forEach((metric) => graphics.append(metricGraphic(metric)));
    panel.append(graphics, node('p', 'scale-note', '每项指标独立尺度，前后共用线性零基线；条长不可跨指标比较。极小值请以标注数值为准，负值或缺失值仅显示数值。'));
  } else bulletList(panel, record.validation?.checks);
  const coverage = record.validation?.coverage;
  if (coverage) {
    const line = node('div', 'coverage');
    line.append(node('span', '', '验证覆盖'), node('strong', '', `${text(coverage.observed)} / ${text(coverage.expected)} ${text(coverage.unit, '')}`));
    panel.append(line);
  }
  body.append(panel);
  const details = node('details', 'details');
  details.append(node('summary', '', '展开诊断、阶段与验证证据'));
  const grid = node('div', 'detail-grid');
  const problem = section('01 / 问题与触发条件');
  [['触发条件', record.problem?.trigger], ['预期', record.problem?.expected], ['实际', record.problem?.observed], ['位置', record.problem?.location], ['根因类别', record.cause?.category]].forEach(([label, value]) => paragraph(problem, label, value));
  bulletList(problem, record.cause?.components);
  const environment = section('02 / 固定版本与修复');
  paragraph(environment, '模型', record.model?.name);
  paragraph(environment, '模型 revision', record.model?.revision);
  paragraph(environment, '硬件', record.environment?.hardware);
  bulletList(environment, libraries(record).map(libraryLabel));
  paragraph(environment, '修复 commit', record.repair?.commit);
  paragraph(environment, '上游状态', record.repair?.upstream_status);
  const stages = section('03 / 修复阶段');
  const stageList = node('ol', 'stages');
  list(record.stages).forEach((stage) => {
    const li = node('li');
    li.append(node('strong', '', text(stage.name)), node('p', '', text(stage.result)));
    stageList.append(li);
  });
  stages.append(stageList.childElementCount ? stageList : node('p', '', '未记录阶段。'));
  const validation = section('04 / 验证范围与证据');
  paragraph(validation, '范围', record.validation?.scope);
  bulletList(validation, record.validation?.checks);
  if (list(record.validation?.remaining).length) {
    validation.append(node('p', '', '待定位与后续验证'));
    bulletList(validation, record.validation.remaining);
  }
  const evidence = node('ul');
  list(record.evidence).forEach((entry) => {
    const url = publicURL(entry.url);
    const li = node('li');
    if (url) {
      const a = node('a', '', text(entry.label, '查看证据'));
      a.href = url;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.setAttribute('aria-label', `${text(entry.label, '查看证据')}（在新标签页打开）`);
      li.append(a);
    } else li.append(node('span', '', `${text(entry.label, '证据')}（无可用公开链接）`));
    evidence.append(li);
  });
  if (evidence.childElementCount) validation.append(evidence);
  if (list(record.notes).length) {
    validation.append(node('p', '', '补充说明'));
    bulletList(validation, record.notes);
  }
  grid.append(problem, environment, stages, validation);
  details.append(grid);
  body.append(details);
  card.append(header, body);
  return card;
}
function renderLibraryGroup(group, groupIndex) {
  const section = node('section', 'library-group');
  section.id = `library-${group.id}`;
  const header = node('div', 'library-group-header');
  const label = node('div');
  label.append(node('p', 'eyebrow', 'LIBRARY GROUP'), node('h3', '', group.title));
  if (group.description) label.append(node('p', 'group-description', group.description));
  const totals = historyAPI.counts([group]);
  header.append(label, node('span', 'group-count', `${totals.fixes} 项修复 · ${totals.verifications} 轮验证`));
  section.append(header);
  group.fixes.forEach((fix, index) => {
    const item = node('details', 'repair-item');
    item.id = `fix-${fix.root.id}`;
    item.open = groupIndex === 0 && index === 0;
    const summary = node('summary', 'issue-summary');
    const title = node('div', 'issue-title');
    title.append(node('h4', '', fix.root.title));
    const metadata = [fix.root.affected_library ? `缺陷所在：${fix.root.affected_library}` : '', `${fix.records.length} 轮验证`].filter(Boolean);
    title.append(node('span', '', metadata.join(' · ')));
    summary.append(title);
    item.append(summary);
    const content = node('div', 'issue-content');
    content.append(renderDiagnosis(fix.root), node('h4', 'verification-heading', '验证历史'));
    fix.records.forEach((record) => content.append(renderCard(record)));
    item.append(content);
    section.append(item);
  });
  return section;
}
function showState(title, description, retry = false) {
  const state = $('state');
  state.replaceChildren(node('h3', '', title), node('p', '', description));
  state.hidden = false;
  if (retry) {
    const button = node('button', '', '重新加载');
    button.type = 'button';
    button.addEventListener('click', load);
    state.append(button);
  }
}
const pageCopy = {
  repairs: { title: ['每一次修复，', '都有证据。'], intro: '从异常到根因，从补丁到验证。跨模型、框架与硬件，记录可复核的精度修复过程。',
    eyebrow: 'REPAIR HISTORY / 精度工程记录', heading: '按库查看修复', noun: '修复',
    labels: ['独立修复', '验证轮次', '库分组', '数据更新'], feet: ['REPAIRS', 'VERIFICATIONS', 'LIBRARY GROUPS', 'LAST UPDATED'],
    method: ['发现偏差 ↗ 定位根因', '实施修复 ↗ 原始负载验证'], caption: '真实记录 · 修复过程 · 验证证据' },
  evaluations: { title: ['跨环境比较，', '留下数值证据。'], intro: '记录推理、SFT 与 RL 的精度采集与比较。查看各侧实际配置、比较条件、指标和覆盖范围。',
    eyebrow: 'EVALUATION HISTORY / 精度工程记录', heading: '按任务查看评测', noun: '评测',
    labels: ['评测实验', '比较组', '硬件平台', '数据更新'], feet: ['EXPERIMENTS', 'COMPARISONS', 'HARDWARE', 'LAST UPDATED'],
    method: ['对齐条件 ↗ 原生采集', '坐标配对 ↗ 输出指标'], caption: '实际配置 · 数值差异 · 覆盖与证据' }
};
function configurePage() {
  const copy = pageCopy[activeKind];
  document.title = `精度${copy.noun}历史 · Precision-Skills`;
  $('hero-kind').textContent = copy.eyebrow;
  $('page-title').replaceChildren(document.createTextNode(copy.title[0]), node('br'), node('span', '', copy.title[1]));
  $('page-intro').textContent = copy.intro;
  const methodText = node('p');
  methodText.append(document.createTextNode(copy.method[0]), node('br'), document.createTextNode(copy.method[1]));
  $('method').replaceChildren(node('span', 'note-index', '01 — METHOD'), methodText, node('span', 'note-caption', copy.caption));
  $('history-title').textContent = copy.heading;
  copy.labels.forEach((value, index) => { $(`stat-label-${index}`).textContent = value; });
  copy.feet.forEach((value, index) => { $(`stat-foot-${index}`).textContent = value; });
  $('task-filter').hidden = activeKind !== 'evaluations';
  $('filters').classList.toggle('evaluation-filters', activeKind === 'evaluations');
  $('filters').setAttribute('aria-label', `筛选${copy.noun}记录`);
  $('search').placeholder = activeKind === 'evaluations' ? '搜索框架、比较条件或指标…' : '搜索问题、根因或修复…';
  for (const kind of ['repairs', 'evaluations']) {
    $(`tab-${kind}`).setAttribute('aria-selected', String(kind === activeKind));
    $(`tab-${kind}`).tabIndex = kind === activeKind ? 0 : -1;
  }
  $('archive-panel').setAttribute('aria-labelledby', `tab-${activeKind}`);
}
function renderEvaluationGroup(group) {
  const element = node('section', 'library-group');
  const header = node('div', 'library-group-header');
  const label = node('div');
  label.append(node('p', 'eyebrow', 'EVALUATION TASK'), node('h3', '', group.title));
  const total = historyAPI.evaluationCounts(group.records);
  header.append(label, node('span', 'group-count', `${total.experiments} 项评测 · ${total.comparisons} 组比较`));
  element.append(header);
  group.records.forEach((record) => element.append(globalThis.PrecisionEvaluations.renderCard(record,
    { node, list, text, paragraph, section, bulletList, publicURL, formatNumber, libraryLabel })));
  return element;
}
function render() {
  const filters = { query: $('search').value, task: $('task').value,
    library: $('library').value, model: $('model').value, hardware: $('hardware').value };
  let visible;
  if (activeKind === 'evaluations') {
    const filtered = historyAPI.filterEvaluations(evaluations, filters);
    $('history').replaceChildren(...historyAPI.evaluationGroups(filtered).map(renderEvaluationGroup));
    const shown = historyAPI.evaluationCounts(filtered), total = historyAPI.evaluationCounts(evaluations);
    $('result-count').textContent = `显示 ${shown.experiments} / ${total.experiments} 项评测 · ${shown.comparisons} / ${total.comparisons} 组比较`;
    visible = filtered.length;
  } else {
    const filtered = historyAPI.filterCatalog(catalog, filters);
    $('history').replaceChildren(...filtered.map(renderLibraryGroup));
    const shown = historyAPI.counts(filtered), total = historyAPI.counts(catalog);
    $('result-count').textContent = `显示 ${shown.groups} 个库分组 · ${shown.fixes} / ${total.fixes} 项修复 · ${shown.verifications} / ${total.verifications} 轮验证`;
    visible = filtered.length;
  }
  $('reset').hidden = !controls.some((id) => $(id).value);
  if (visible) $('state').hidden = true;
  else showState(`没有匹配的${pageCopy[activeKind].noun}`, '尝试其他关键词，或清除筛选以查看全部记录。');
}
function resetFilters() {
  $('filters').reset();
  if (archive) render();
}
function selectTab(kind) {
  if (kind === activeKind) return;
  activeKind = kind;
  $('filters').reset();
  configurePage();
  load();
}
function followHash() {
  const route = historyAPI.routeForHash(location.hash);
  if (!route) return;
  if (route.kind !== activeKind) { selectTab(route.kind); return; }
  if (!archive || !route.id || ['repairs', 'evaluations'].includes(route.id)) return;
  const id = route.id;
  const known = activeKind === 'evaluations' ? evaluations.some((record) => `evaluation-${record.id}` === id) :
    repairs.some((record) => `repair-${record.id}` === id) || catalog.some((group) =>
      `library-${group.id}` === id || group.fixes.some((fix) => `fix-${fix.root.id}` === id));
  if (!known) return;
  if (!document.getElementById(id)) resetFilters();
  const card = document.getElementById(id);
  if (card) {
    openAncestors(card);
    if (card.tagName === 'DETAILS') card.open = true;
    else if (card.querySelector('details')) card.querySelector('details').open = true;
    card.scrollIntoView({ block: 'start' });
  }
}
async function load() {
  pendingRequest?.abort();
  const serial = ++requestSerial, kind = activeKind, copy = pageCopy[kind];
  archive = null;
  $('download').disabled = true;
  $('history').replaceChildren();
  ['total-count', 'verification-count', 'group-count', 'updated-at'].forEach((id) => { $(id).textContent = '—'; });
  $('result-count').textContent = `正在读取${copy.noun}档案…`;
  showState('正在加载记录', `准备${copy.noun}指标与验证证据。`);
  const controller = new AbortController();
  pendingRequest = controller;
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(`./data/${kind}.json`, { signal: controller.signal });
    if (!response.ok) throw new Error('HTTP');
    const data = await response.json();
    if (serial !== requestSerial) return;
    if (data.schema_version !== 1 || !Array.isArray(data[kind])) throw new Error('schema');
    const ids = new Set();
    for (const record of data[kind]) {
      if (!record || typeof record.id !== 'string' || !record.id.trim() || ids.has(record.id)) throw new Error('id');
      ids.add(record.id);
    }
    let options;
    if (kind === 'evaluations') {
      evaluations = historyAPI.filterEvaluations(data.evaluations);
      const totals = historyAPI.evaluationCounts(evaluations);
      $('total-count').textContent = totals.experiments;
      $('verification-count').textContent = totals.comparisons;
      $('group-count').textContent = totals.hardware;
      options = [['library', evaluations.flatMap(historyAPI.evaluationLibraries)], ['model', evaluations.map((r) => r.model.name)],
        ['hardware', evaluations.flatMap((r) => r.sides.map((s) => s.hardware))]];
    } else {
      catalog = historyAPI.buildCatalog(data);
      repairs = [...data.repairs].sort((a, b) => String(b.date || '').localeCompare(String(a.date || '')));
      const totals = historyAPI.counts(catalog);
      $('total-count').textContent = totals.fixes;
      $('verification-count').textContent = totals.verifications;
      $('group-count').textContent = totals.groups;
      options = [['library', catalog.flatMap((g) => g.components)], ['model', repairs.map((r) => r.model?.name)], ['hardware', repairs.map((r) => r.environment?.hardware)]];
    }
    archive = data;
    $('updated-at').textContent = text(data.updated_at, '未记录');
    options.forEach(([id, values]) => {
      const select = $(id);
      select.replaceChildren(select.options[0]);
      [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b)).forEach((value) => {
        const option = node('option', '', value);
        option.value = value;
        select.append(option);
      });
    });
    $('download').disabled = false;
    render();
    followHash();
  } catch {
    if (serial !== requestSerial) return;
    archive = null;
    $('history').replaceChildren();
    $('result-count').textContent = '档案暂时不可用';
    showState(`暂时无法加载${copy.noun}档案`, '请稍后重试。', true);
  } finally { clearTimeout(timeout); }
}
$('filters').addEventListener('submit', (event) => event.preventDefault());
controls.forEach((id) => $(id).addEventListener(id === 'search' ? 'input' : 'change', () => { if (archive) render(); }));
$('reset').addEventListener('click', resetFilters);
for (const kind of ['repairs', 'evaluations']) $('tab-' + kind).addEventListener('click', () => {
  if (location.hash !== '#' + kind) location.hash = kind;
});
document.querySelector('[role="tablist"]').addEventListener('keydown', (event) => {
  const keys = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
  if (!keys.includes(event.key)) return;
  event.preventDefault();
  const kind = event.key === 'Home' ? 'repairs' : event.key === 'End' ? 'evaluations' :
    event.target.id === 'tab-repairs' ? 'evaluations' : 'repairs';
  $('tab-' + kind).focus();
  $('tab-' + kind).click();
});
window.addEventListener('hashchange', followHash);
$('download').addEventListener('click', () => {
  if (!archive) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(archive, null, 2)], { type: 'application/json;charset=utf-8' }));
  const link = node('a');
  link.href = url;
  link.download = `precision-skills-${activeKind}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
activeKind = historyAPI.routeForHash(location.hash)?.kind || 'repairs';
configurePage();
load();
