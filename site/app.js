'use strict';

const $ = (id) => document.getElementById(id);
const controls = ['search', 'library', 'model', 'hardware'];
let archive = null;
let repairs = [];

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
  element.append(node('h4', '', title));
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
function renderCard(record) {
  const card = node('article', 'repair-card');
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
  link.addEventListener('click', () => { card.querySelector('details').open = true; });
  const mobileDate = node('time', 'mobile-date', text(record.date));
  if (/^\d{4}-\d{2}-\d{2}$/.test(record.date)) mobileDate.dateTime = record.date;
  top.append(link, mobileDate);
  const title = node('h3', '', text(record.title));
  heading.append(top, title, node('p', 'card-summary', text(record.problem?.summary)));
  const tags = node('div', 'tags');
  [record.model?.name, record.environment?.hardware, ...libraries(record).map(libraryLabel)].filter(Boolean).forEach((tag) => tags.append(node('span', '', tag)));
  heading.append(tags);
  header.append(rail, heading);
  const body = node('div', 'card-body');
  const diagnosis = node('div', 'diagnosis');
  [['根因 / CAUSE', record.cause?.summary], ['修复 / REPAIR', record.repair?.summary]].forEach(([label, summary]) => {
    const block = node('div');
    block.append(node('span', 'small-label', label), node('p', '', text(summary)));
    diagnosis.append(block);
  });
  body.append(diagnosis);
  const metrics = list(record.validation?.metrics);
  const panel = node('section', 'metric-panel');
  const panelHeading = node('div', 'metric-heading');
  panelHeading.append(node('h4', '', '验证指标'));
  const legend = node('div', 'legend');
  legend.setAttribute('aria-hidden', 'true');
  legend.append(node('span', '', '修复前'), node('span', '', '修复后'));
  panelHeading.append(legend);
  panel.append(panelHeading);
  if (metrics.length) {
    const graphics = node('div', 'metrics');
    metrics.forEach((metric) => graphics.append(metricGraphic(metric)));
    panel.append(graphics, node('p', 'scale-note', '每项指标独立尺度，前后共用线性零基线；条长不可跨指标比较。极小值请以标注数值为准，负值或缺失值仅显示数值。'));
  } else panel.append(node('p', 'scale-note', '本记录未提供可展示的前后数值。'));
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
  const validation = section('04 / 验证边界与证据');
  paragraph(validation, '范围', record.validation?.scope);
  bulletList(validation, record.validation?.checks);
  if (list(record.validation?.remaining).length) {
    paragraph(validation, '仍待确认', '以下事项不包含在已完成验证中');
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
function render() {
  const query = $('search').value.trim().toLocaleLowerCase();
  const filtered = repairs.filter((record) =>
    (!query || JSON.stringify(record).toLocaleLowerCase().includes(query)) &&
    (!$('library').value || libraries(record).some((lib) => lib.name === $('library').value)) &&
    (!$('model').value || record.model?.name === $('model').value) &&
    (!$('hardware').value || record.environment?.hardware === $('hardware').value));
  $('history').replaceChildren(...filtered.map(renderCard));
  $('result-count').textContent = `显示 ${filtered.length} / ${repairs.length} 条修复记录 · 按日期倒序`;
  $('reset').hidden = !controls.some((id) => $(id).value);
  if (filtered.length) $('state').hidden = true;
  else showState(repairs.length ? '没有匹配的修复记录' : '档案尚无修复记录', repairs.length ? '尝试其他关键词，或清除筛选以查看全部记录。' : '首条修复记录发布后，将在这里展示诊断与验证证据。');
}
function resetFilters() {
  $('filters').reset();
  if (archive) render();
}
function followHash() {
  if (!archive || !location.hash) return;
  let id;
  try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
  if (!repairs.some((record) => `repair-${record.id}` === id)) return;
  if (!document.getElementById(id)) resetFilters();
  const card = document.getElementById(id);
  if (card) {
    card.querySelector('details').open = true;
    card.scrollIntoView({ block: 'start' });
  }
}
async function load() {
  $('download').disabled = true;
  $('result-count').textContent = '正在读取修复档案…';
  showState('正在加载记录', '准备修复过程与验证证据。');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch('./data/repairs.json', { signal: controller.signal });
    if (!response.ok) throw new Error('HTTP');
    const data = await response.json();
    if (data.schema_version !== 1 || !Array.isArray(data.repairs)) throw new Error('schema');
    const ids = new Set();
    for (const record of data.repairs) {
      if (!record || typeof record.id !== 'string' || !record.id.trim() || ids.has(record.id)) throw new Error('id');
      ids.add(record.id);
    }
    archive = data;
    repairs = [...data.repairs].sort((a, b) => String(b.date || '').localeCompare(String(a.date || '')));
    $('total-count').textContent = repairs.length;
    $('validated-count').textContent = repairs.filter((record) => record.status === 'validated').length;
    $('library-count').textContent = new Set(repairs.flatMap((record) => libraries(record).map((lib) => lib.name)).filter(Boolean)).size;
    $('updated-at').textContent = text(data.updated_at, '未记录');
    [['library', repairs.flatMap((record) => libraries(record).map((lib) => lib.name))], ['model', repairs.map((record) => record.model?.name)], ['hardware', repairs.map((record) => record.environment?.hardware)]].forEach(([id, values]) => {
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
    archive = null;
    repairs = [];
    $('history').replaceChildren();
    ['total-count', 'validated-count', 'library-count', 'updated-at'].forEach((id) => { $(id).textContent = '—'; });
    $('result-count').textContent = '档案暂时不可用';
    showState('暂时无法加载修复档案', '数据可能尚未发布，或网络暂时不可用。请稍后重试。', true);
  } finally { clearTimeout(timeout); }
}
$('filters').addEventListener('submit', (event) => event.preventDefault());
controls.forEach((id) => $(id).addEventListener(id === 'search' ? 'input' : 'change', () => { if (archive) render(); }));
$('reset').addEventListener('click', resetFilters);
window.addEventListener('hashchange', followHash);
$('download').addEventListener('click', () => {
  if (!archive) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(archive, null, 2)], { type: 'application/json;charset=utf-8' }));
  const link = node('a');
  link.href = url;
  link.download = 'precision-skills-repairs.json';
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
load();
