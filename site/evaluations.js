/* Evaluation cards show peer comparisons, native settings and coordinate evidence. */
(function (global) {
  'use strict';
  const finite = (value) => typeof value === 'number' && Number.isFinite(value);

  // Normalize before subtracting so a signed domain such as [-1e308, 1e308]
  // remains drawable. No visual minimum is applied to zero or very small values.
  function barGeometry(value, domain) {
    if (!finite(value) || !Array.isArray(domain) || domain.length !== 2 || !domain.every(finite)) return null;
    const [minimum, maximum] = domain;
    if (minimum > 0 || maximum < 0 || minimum > maximum || value < minimum || value > maximum) return null;
    const magnitude = Math.max(Math.abs(minimum), Math.abs(maximum));
    if (!magnitude) return { left: 0, width: 0, zero: 0 };
    const low = minimum / magnitude;
    const span = maximum / magnitude - low;
    const position = (number) => Math.max(0, Math.min(100, (number / magnitude - low) / span * 100));
    const zero = position(0);
    const end = position(value);
    return { left: Math.min(zero, end), width: Math.abs(end - zero), zero };
  }

  function metricFamily(metric) {
    const identity = `${metric.id || ''} ${metric.label || ''}`.toLowerCase();
    if (/cosine|余弦/.test(identity)) return 'cosine';
    if (/agreement|overlap|一致率|重合率/.test(identity)) return 'fraction';
    if (/(?:^|[\s-])js(?:[\s-]|$)|jensen.shannon/.test(identity)) return 'js';
    if (/error|(?:^|[\s-])(?:mae|rmse)(?:[\s-]|$)|max-abs|abs-diff|绝对差|绝对误差|误差/.test(identity) ||
        /^(?:target-max|target-p99(?:-nat)?|p99-nat)$/.test(metric.id || '')) return 'error';
    return null;
  }

  // Domains are local to the selected comparison. Side values have a shared
  // zero baseline; scalar error and JS groups share a linear domain by unit.
  function chartModel(metric, comparison) {
    const sideIds = Array.isArray(comparison.side_ids) ? comparison.side_ids : [];
    const rowsFor = (domain, values) => values.map((row) => ({ ...row, geometry: barGeometry(row.value, domain) }));
    if (metric.side_values) {
      const rows = sideIds.map((id) => ({ id, value: metric.side_values[id] }));
      if (!rows.length || !rows.every((row) => finite(row.value))) {
        return { kind: 'none', scale: 'none', domain: null, rows, reason: 'invalid-sides' };
      }
      const domain = [Math.min(0, ...rows.map((row) => row.value)), Math.max(0, ...rows.map((row) => row.value))];
      return { kind: 'sides', scale: 'peer', domain, rows: rowsFor(domain, rows) };
    }
    const family = metricFamily(metric);
    const single = [{ id: null, value: metric.value }];
    if (!finite(metric.value)) return { kind: 'none', scale: 'none', domain: null, rows: single, reason: 'invalid-value' };
    if (family === 'cosine' || family === 'fraction') {
      const dimensionless = metric.unit === '1' || family === 'fraction' && metric.unit === 'fraction';
      if (!dimensionless) return { kind: 'none', scale: 'none', domain: null, rows: single, reason: 'unknown-range' };
      const domain = family === 'cosine' ? [-1, 1] : [0, 1];
      if (metric.value < domain[0] || metric.value > domain[1]) {
        return { kind: 'none', scale: 'none', domain: null, rows: single, reason: 'out-of-range' };
      }
      return { kind: family, scale: 'fixed', domain, rows: rowsFor(domain, single) };
    }
    if ((family === 'error' || family === 'js') && metric.value >= 0) {
      const peers = (comparison.metrics || []).filter((peer) => !peer.side_values && peer.unit === metric.unit &&
        metricFamily(peer) === family && finite(peer.value) && peer.value >= 0);
      const domain = [0, Math.max(metric.value, ...peers.map((peer) => peer.value))];
      return { kind: family === 'js' ? 'divergence' : 'error', family, scale: 'comparison-unit', domain,
        rows: rowsFor(domain, single) };
    }
    return { kind: 'none', scale: 'none', domain: null, rows: single,
      reason: family === 'error' || family === 'js' ? 'negative-value' : 'unknown-range' };
  }

  function coverageModel(row) {
    if (!finite(row.observed) || !finite(row.expected) || row.observed < 0 || row.expected < 0) {
      return { ratio: null, width: null, drawable: false, state: 'invalid' };
    }
    if (!row.expected) return { ratio: null, width: null, drawable: false, state: row.observed ? 'invalid' : 'empty' };
    const ratio = row.observed / row.expected;
    return { ratio, width: Math.min(1, ratio) * 100, drawable: true, state: row.observed > row.expected ? 'over' : 'within' };
  }

  function renderCard(record, helpers) {
    const { node, list, text, paragraph, section, bulletList, publicURL, formatNumber, libraryLabel } = helpers;
    const sideLabel = (id) => record.sides.find((side) => side.id === id).label;
    const unitLabel = (unit) => ({ '1': '无量纲', gradient: '梯度值', parameter: '参数值', arrays: '数组' }[unit] || unit);
    const card = node('article', 'repair-card evaluation-card');
    card.id = `evaluation-${record.id}`;
    const header = node('div', 'card-header');
    const date = node('time', 'evaluation-date', record.date);
    date.dateTime = record.date;
    const heading = node('div', 'card-heading');
    const top = node('div', 'card-topline');
    top.append(node('span', `pill${record.status === 'completed' ? ' validated' : ''}`,
      record.status === 'completed' ? '采集与比较完成' : '部分完成'));
    top.append(node('span', `pill${record.acceptance == null ? ' pending' : ''}`,
      record.acceptance == null ? '验收 · 待用户判定' : `验收 · ${text(record.acceptance.status || record.acceptance)}`));
    const link = node('a', 'permalink', '条目链接 ↗');
    link.href = `#${card.id}`;
    top.append(link, date);
    heading.append(top, node('h3', '', record.title), node('p', 'card-summary', record.summary));
    const tags = node('div', 'tags');
    [record.model.name, ...new Set(record.sides.map((side) => side.hardware))].forEach((tag) => tags.append(node('span', '', tag)));
    heading.append(tags);
    header.append(heading);
    const body = node('div', 'card-body');
    const selection = node('div', 'comparison-picker');
    const label = node('label', '', '比较范围');
    const select = node('select');
    select.id = `comparison-${record.id}`;
    label.htmlFor = select.id;
    record.comparisons.forEach((comparison) => {
      const option = node('option', '', comparison.title);
      option.value = comparison.id;
      select.append(option);
    });
    selection.append(label, select);
    const panel = node('section', 'metric-panel evaluation-metrics');
    panel.setAttribute('aria-label', '评测指标');
    const coordinates = node('div');
    function showComparison() {
      const comparison = record.comparisons.find((item) => item.id === select.value);
      panel.replaceChildren(node('h4', 'comparison-title', comparison.side_ids.map(sideLabel).join(' ↔ ')),
        node('p', 'comparison-scope', comparison.scope));
      panel.append(node('p', 'evaluation-chart-note', '灰 / 绿仅区分比较侧；条长表示数值比例，不表示好坏或验收阈值。'));
      const metrics = node('dl', 'evaluation-metric-grid');
      comparison.metrics.forEach((metric) => {
        const item = node('div', 'evaluation-metric');
        item.dataset.metricId = metric.id;
        const value = node('dd', 'evaluation-value', formatNumber(metric.value));
        value.append(node('span', 'metric-unit', `${metric.side_values ? '对称差 · ' : ''}${unitLabel(metric.unit)}`));
        item.append(node('dt', '', metric.label), value);
        const model = chartModel(metric, comparison);
        const chart = node('dd', 'evaluation-chart');
        chart.dataset.kind = model.kind;
        if (model.domain) {
          chart.dataset.domainMin = String(model.domain[0]);
          chart.dataset.domainMax = String(model.domain[1]);
          const values = node('ul', model.kind === 'sides' ? 'side-values' : 'evaluation-scalar-values');
          model.rows.forEach((row, index) => {
            const line = node('li', `evaluation-bar-row${model.kind === 'sides' && index ? ' evaluation-side-b' : ''}`);
            const label = model.kind === 'sides' ? sideLabel(row.id) : '比较值';
            line.append(node('span', 'evaluation-bar-label', label),
              node('span', 'evaluation-bar-value', formatNumber(row.value)));
            const track = node('div', 'track evaluation-track');
            track.setAttribute('aria-hidden', 'true');
            const zero = node('span', 'evaluation-zero');
            zero.style.left = `${row.geometry.zero}%`;
            const bar = node('span', `bar evaluation-bar${model.kind === 'sides' && index ? ' after' : ''}`);
            bar.dataset.value = String(row.value);
            if (row.id !== null) bar.dataset.sideId = row.id;
            bar.style.left = `${row.geometry.left}%`;
            bar.style.width = `${row.geometry.width}%`;
            track.append(bar, zero);
            line.append(track);
            values.append(line);
          });
          chart.append(values);
          const [minimum, maximum] = model.domain;
          const axis = node('div', 'evaluation-axis');
          axis.setAttribute('aria-hidden', 'true');
          axis.append(node('span', '', formatNumber(minimum)), node('span', '', formatNumber(maximum)));
          chart.append(axis);
          const scale = model.scale === 'peer' ? '两侧共同线性域' : model.scale === 'fixed' ?
            (model.kind === 'cosine' ? '余弦范围' : '比例范围') :
            `本比较${model.kind === 'divergence' ? ' JS 同单位' : '同单位误差'}共同线性域`;
          const zeroNote = minimum === 0 && maximum === 0 ? '；全部为 0' : minimum < 0 && maximum > 0 ? '；竖线为 0' : '';
          chart.append(node('p', 'evaluation-scale', `${scale} [${formatNumber(minimum)}, ${formatNumber(maximum)}]${zeroNote}`));
        } else {
          if (metric.side_values) {
            const values = node('ul', 'side-values');
            model.rows.forEach((row) => values.append(node('li', '', `${sideLabel(row.id)}：${formatNumber(row.value)}`)));
            chart.append(values);
          }
          chart.append(node('p', 'evaluation-scale', model.reason === 'out-of-range' ? '超出定义域，仅显示原值' : '未设展示范围，仅显示原值'));
        }
        item.append(chart);
        metrics.append(item);
      });
      panel.append(metrics);
      const coverage = node('div', 'evaluation-coverage');
      comparison.coverage.forEach((row) => {
        const line = node('div', 'coverage');
        const heading = node('div', 'evaluation-coverage-heading');
        heading.append(node('span', '', row.label), node('strong', '', `${row.observed} / ${row.expected} ${row.unit}`));
        line.append(heading);
        const model = coverageModel(row);
        const track = node('div', 'track evaluation-coverage-track');
        track.dataset.observed = String(row.observed);
        track.dataset.expected = String(row.expected);
        track.dataset.state = model.state;
        track.setAttribute('aria-hidden', 'true');
        if (model.drawable) {
          const bar = node('span', 'bar evaluation-coverage-bar');
          bar.style.width = `${model.width}%`;
          track.append(bar);
        }
        line.append(track, node('span', 'evaluation-coverage-note', model.state === 'over' ? '观测 / 预期 · 已超出预期计数' :
          model.state === 'empty' ? '观测 / 预期 · 预期为 0，不计算比例' : model.state === 'invalid' ? '观测 / 预期 · 不计算比例' : '观测 / 预期'));
        coverage.append(line);
      });
      panel.append(coverage);
      coordinates.replaceChildren();
      comparison.metrics.filter((metric) => metric.coordinate).forEach((metric) => {
        const item = node('p');
        item.append(node('strong', '', `${metric.label}：`), node('code', '', JSON.stringify(metric.coordinate)));
        coordinates.append(item);
      });
      if (!coordinates.childElementCount) coordinates.append(node('p', '', '该比较未在摘要中列出坐标，详见归档证据。'));
    }
    select.addEventListener('change', showComparison);
    showComparison();
    const details = node('details', 'details');
    details.append(node('summary', '', '展开配置、最大误差坐标与证据'));
    const grid = node('div', 'detail-grid');
    const conditions = section('01 / 比较条件');
    paragraph(conditions, '模型 revision', record.model.revision);
    conditions.append(node('p', 'small-label', '对齐条件'));
    bulletList(conditions, record.common);
    conditions.append(node('p', 'small-label', '比较变量'));
    bulletList(conditions, record.differences);
    const environments = section('02 / 各侧实际配置');
    record.sides.forEach((side) => {
      environments.append(node('p', 'side-heading', side.label));
      paragraph(environments, '硬件', side.hardware);
      bulletList(environments, side.libraries.map(libraryLabel));
      Object.entries(side.settings).forEach(([key, value]) => paragraph(environments, key,
        value !== null && typeof value === 'object' ? JSON.stringify(value) : value));
    });
    const maximum = section('03 / 当前比较的最大误差坐标');
    maximum.append(coordinates);
    const evidence = section('04 / 归档与复核');
    bulletList(evidence, record.notes);
    paragraph(evidence, '归档 revision', record.archive.revision);
    paragraph(evidence, '归档前缀', record.archive.prefix);
    const links = node('ul');
    record.evidence.forEach((entry) => {
      const url = publicURL(entry.url);
      const li = node('li');
      if (url) {
        const anchor = node('a', '', entry.label);
        anchor.href = url;
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
        li.append(anchor);
      } else li.append(node('span', '', entry.label));
      links.append(li);
    });
    evidence.append(links);
    grid.append(conditions, environments, maximum, evidence);
    details.append(grid);
    body.append(selection, panel, details);
    card.append(header, body);
    return card;
  }
  const api = { renderCard, chartModel, barGeometry, coverageModel };
  if (typeof module === 'object' && module.exports) module.exports = api;
  global.PrecisionEvaluations = api;
})(globalThis);
