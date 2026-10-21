/* Evaluation cards show peer comparisons, native settings and coordinate evidence. */
(function (global) {
  'use strict';
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
      const metrics = node('dl', 'evaluation-metric-grid');
      comparison.metrics.forEach((metric) => {
        const item = node('div', 'evaluation-metric');
        const value = node('dd', 'evaluation-value', formatNumber(metric.value));
        value.append(node('span', 'metric-unit', unitLabel(metric.unit)));
        item.append(node('dt', '', metric.label), value);
        if (metric.side_values) {
          const values = node('ul', 'side-values');
          comparison.side_ids.forEach((id) => values.append(node('li', '', `${sideLabel(id)}：${formatNumber(metric.side_values[id])}`)));
          value.append(values);
        }
        metrics.append(item);
      });
      panel.append(metrics);
      const coverage = node('div', 'evaluation-coverage');
      comparison.coverage.forEach((row) => {
        const line = node('div', 'coverage');
        line.append(node('span', '', row.label), node('strong', '', `${row.observed} / ${row.expected} ${row.unit}`));
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
  global.PrecisionEvaluations = { renderCard };
})(globalThis);
