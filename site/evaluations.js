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

  // Draw only a genuine, unequal pair. Summary statistics and counts stay numerical.
  function chartModel(metric, comparison) {
    const sideIds = Array.isArray(comparison.side_ids) ? comparison.side_ids : [];
    const none = (rows, reason) => ({ kind: 'none', scale: 'none', domain: null, rows, reason });
    if (!metric.side_values) return none([{ id: null, value: metric.value }], 'summary');
    const rows = sideIds.map((id) => ({ id, value: metric.side_values[id] }));
    if (sideIds.length !== 2 || new Set(sideIds).size !== 2 || !rows.every((row) => finite(row.value))) {
      return none(rows, 'invalid-sides');
    }
    if (rows[0].value === rows[1].value) return none(rows, 'equal-sides');
    const domain = [Math.min(0, ...rows.map((row) => row.value)), Math.max(0, ...rows.map((row) => row.value))];
    return { kind: 'sides', scale: 'peer', domain,
      rows: rows.map((row) => ({ ...row, geometry: barGeometry(row.value, domain) })) };
  }

  function renderCard(record, helpers) {
    const { node, list, text, paragraph, section, bulletList, publicURL, formatNumber, libraryLabel,
      modelLabel = (entry) => entry.model.name,
      hardwareLabels = (entry) => [...new Set(entry.sides.map((side) => side.hardware))] } = helpers;
    const sideLabel = (id) => record.sides.find((side) => side.id === id).label;
    const unitLabel = (unit) => ({ '1': '无量纲', gradient: '梯度值', parameter: '参数值', arrays: '数组' }[unit] || unit);
    const card = node('details', 'repair-card repair-item evaluation-card evaluation-item');
    card.id = `evaluation-${record.id}`;
    const header = node('summary', 'card-header issue-summary evaluation-summary');
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
    link.addEventListener('click', (event) => {
      event.stopPropagation();
      card.open = true;
      details.open = true;
    });
    top.append(link, date);
    const title = global.EvaluationPresentation?.titleFor(record, modelLabel(record)) || record.title;
    heading.append(top, node('h3', '', title),
      node('p', 'card-summary', `${record.comparisons.length} 组比较 · ${record.sides.length} 组配置`));
    const tags = node('div', 'tags');
    [modelLabel(record), ...hardwareLabels(record)].forEach((tag) => tags.append(node('span', '', tag)));
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
      if (comparison.metrics.some((metric) => chartModel(metric, comparison).kind === 'sides')) {
        panel.append(node('p', 'evaluation-chart-note', '灰 / 绿对应两侧配置；同项共用零基线，精确数值见标注。'));
      }
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
          const values = node('ul', 'side-values');
          model.rows.forEach((row, index) => {
            const line = node('li', `evaluation-bar-row${index ? ' evaluation-side-b' : ''}`);
            const label = sideLabel(row.id);
            line.append(node('span', 'evaluation-bar-label', label),
              node('span', 'evaluation-bar-value', formatNumber(row.value)));
            const track = node('div', 'track evaluation-track');
            track.setAttribute('aria-hidden', 'true');
            const zero = node('span', 'evaluation-zero');
            zero.style.left = `${row.geometry.zero}%`;
            const bar = node('span', `bar evaluation-bar${index ? ' after' : ''}`);
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
          const scale = '两侧共同线性域';
          const zeroNote = minimum === 0 && maximum === 0 ? '；全部为 0' : minimum < 0 && maximum > 0 ? '；竖线为 0' : '';
          chart.append(node('p', 'evaluation-scale', `${scale} [${formatNumber(minimum)}, ${formatNumber(maximum)}]${zeroNote}`));
        } else {
          if (metric.side_values) {
            const values = node('ul', 'side-values evaluation-pair-values');
            model.rows.forEach((row) => {
              const line = node('li');
              line.append(node('span', 'evaluation-pair-label', sideLabel(row.id)), node('span', 'evaluation-pair-value', formatNumber(row.value)));
              values.append(line);
            });
            chart.append(values);
          }
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
        coverage.append(line);
      });
      panel.append(coverage);
      coordinates.replaceChildren();
      comparison.metrics.filter((metric) => metric.coordinate).forEach((metric) => {
        const item = node('p');
        item.append(node('strong', '', metric.label),
          global.PrecisionEvaluationSettings.renderSettings(metric.coordinate, { node }));
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
    paragraph(conditions, '原始记录标题', record.title);
    paragraph(conditions, '记录模型名称', record.model.name);
    paragraph(conditions, '模型 revision', record.model.revision);
    conditions.append(node('p', 'small-label', '对齐条件'));
    bulletList(conditions, record.common);
    conditions.append(node('p', 'small-label', '比较变量'));
    bulletList(conditions, record.differences);
    const environments = section('02 / 各侧实际配置');
    record.sides.forEach((side, index) => {
      const configuration = node('details', 'evaluation-configuration-side');
      configuration.open = index === 0;
      configuration.append(node('summary', 'side-heading', side.label));
      paragraph(configuration, '硬件', side.hardware);
      bulletList(configuration, side.libraries.map(libraryLabel));
      configuration.append(global.PrecisionEvaluationSettings.renderSettings(side.settings, { node }));
      environments.append(configuration);
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
    body.append(node('p', 'card-summary evaluation-intro', record.summary), selection, panel, details);
    card.append(header, body);
    return card;
  }
  const api = { renderCard, chartModel, barGeometry };
  if (typeof module === 'object' && module.exports) module.exports = api;
  global.PrecisionEvaluations = api;
})(globalThis);
