/* Survey evidence is grouped by model without conflating comparison protocols. */
(function (global) {
  'use strict';
  const list = (value) => Array.isArray(value) ? value : [];
  const finite = (value) => typeof value === 'number' && Number.isFinite(value);
  const kindLabels = {
    'cross-backend': 'GPU ↔ NPU',
    'native-reference': '原生 ↔ 参考',
    'within-backend': '平台内对比',
    'operator-reference': '算子 ↔ 参考',
    'reported-diagnostic': '诊断记录'
  };
  const kindNotes = {
    'cross-backend': '两侧硬件、精度与并行设置请见配置；该结果不自动归因于硬件。',
    'native-reference': '同一平台的原生实现与参考路径比较；不是 GPU 与 NPU 的直接差值。',
    'within-backend': '同一平台内的执行配置比较，保留各自的控制条件。',
    'operator-reference': '算子级参考对比；范围以该记录为准，不代表完整模型误差。',
    'reported-diagnostic': '报告中的诊断证据；不与配对比较合并计算。'
  };
  function modelLabel(record) {
    return typeof record.model === 'string' ? record.model : record.model?.label || record.model?.name || record.model_id || '模型未记录';
  }
  const kindLabel = (kind) => kindLabels[kind] || kind || '类型未记录';
  const platformLabels = (record) => record.platform ? [record.platform] : [];
  function libraries(record) {
    return [...new Set([record.side_a, record.side_b].flatMap((side) => {
      const value = side?.libraries;
      return Array.isArray(value) ? value.map((library) => typeof library === 'string' ? library : library.name).filter(Boolean) :
        value && typeof value === 'object' ? Object.keys(value) : [];
    }))];
  }
  const sorted = (values) => [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b));
  function facets(records) {
    return { models: sorted(records.map(modelLabel)), platforms: sorted(records.flatMap(platformLabels)),
      libraries: sorted(records.flatMap(libraries)), kinds: sorted(records.map((record) => record.comparison_kind)) };
  }
  function filterRecords(records, filters = {}) {
    const query = String(filters.query || '').trim().toLocaleLowerCase();
    return records.filter((record) =>
      (!filters.model || modelLabel(record) === filters.model) &&
      (!filters.hardware || platformLabels(record).includes(filters.hardware)) &&
      (!filters.library || libraries(record).includes(filters.library)) &&
      (!filters.comparison_kind || record.comparison_kind === filters.comparison_kind) &&
      (!query || JSON.stringify(record).toLocaleLowerCase().includes(query) || kindLabel(record.comparison_kind).toLocaleLowerCase().includes(query)));
  }
  const visibleGroups = new Map();
  const kindOrder = ['cross-backend', 'native-reference', 'within-backend', 'operator-reference', 'reported-diagnostic'];
  function revealRecord(id) {
    for (const group of visibleGroups.values()) {
      const index = group.records.findIndex((record) => `difference-${record.id}` === id);
      if (index >= 0) { group.showPage(Math.floor(index / group.pageSize)); return true; }
    }
    return false;
  }
  function groups(records) {
    const result = new Map();
    records.forEach((record) => {
      const label = modelLabel(record);
      if (!result.has(label)) result.set(label, { label, records: [] });
      result.get(label).records.push(record);
    });
    return [...result.values()].map((group) => ({ ...group, records: [...group.records].sort((a, b) => kindOrder.indexOf(a.comparison_kind) - kindOrder.indexOf(b.comparison_kind)) }));
  }
  const metadataMetric = (metric) => /^(?:seed|samples|sample_count|bootstrap_replicates|count|cells|covered_cells|missing_cells|positions|n|num_.*|.*_id|.*_index|index|row|column|position|token|token_id|prompt_length|continuation_length|length|batch_size|vocab_size|layer|layer_index|rank|world_size|tp|pp|ep)$/.test(metric.key || '');
  const metricNames = {
    'target-mae': '目标 Logprob 平均绝对误差', 'target-p99': '目标 Logprob P99 绝对误差', 'target-max': '目标 Logprob 最大绝对误差',
    'nll-gap': '平均 NLL 绝对差异', 'nll-abs-diff': '目标 NLL 绝对差异', 'top1-agreements': 'Top-1 一致位置数',
    equal_input_delta_nat_per_token: '等输入 ΔNLL', mae_nat: 'Logprob 平均绝对误差', p99_nat: 'Logprob P99 绝对误差', max_abs_nat: 'Logprob 最大绝对误差',
    top1_equal: 'Top-1 一致位置数', gpu_rmse: 'GPU 参考 RMSE', npu_rmse: 'NPU 参考 RMSE'
  };
  const metricLabel = (metric) => metricNames[metric.key] || metric.label || metric.key;
  function metricKey(metric, record = {}) {
    return JSON.stringify([metric.key || metric.label || '', metric.unit || '', record.comparison_kind || '', record.source?.path || '', record.stage || '']);
  }
  function metricOptions(records) {
    const options = new Map();
    records.forEach((record) => list(record.metrics).forEach((metric) => {
      if (!finite(metric.value) || metadataMetric(metric) || !metric.unit || metric.unit === 'recorded units') return;
      const key = metricKey(metric, record);
      if (!options.has(key)) options.set(key, { key, label: metricLabel(metric), unit: metric.unit || '', count: 0, kind: record.comparison_kind || '', stage: record.stage || '', source: record.source?.path || '' });
      options.get(key).count += 1;
    }));
    const priority = (option) => /mae|mean.*abs|平均.*绝对/i.test(option.label) ? 0 : /max.*abs|最大.*绝对/i.test(option.label) ? 1 : /top.?1|p99/i.test(option.label) ? 2 : 3;
    return [...options.values()].sort((a, b) => priority(a) - priority(b) || b.count - a.count || a.label.localeCompare(b.label));
  }
  function compactNumber(value) {
    if (!finite(value)) return value === null || value === undefined ? '未记录' : String(value);
    if (value === 0) return '0';
    const magnitude = Math.abs(value);
    return magnitude < 0.001 || magnitude >= 10000 ? value.toExponential(3) : String(Number(value.toPrecision(5)));
  }
  function representativeMetrics(record) {
    const metrics = list(record.metrics).filter((metric) => finite(metric.value) && !metadataMetric(metric));
    const preferred = metrics.filter((metric) => /mae|mean.*abs|max.*abs|top.?1|p99|jensen|cosine|equal_input_delta|nll.*(?:gap|diff)/i.test(metric.key || metric.label));
    return (preferred.length ? preferred : metrics).slice(0, 3);
  }
  function renderArchive(archive, records, helpers) {
    const { node, text, publicURL, formatNumber } = helpers;
    visibleGroups.clear();
    const unitLabel = (unit) => unit === '1' ? '无量纲' : unit || '单位未记录';
    const exactValue = (value) => finite(value) ? formatNumber(value) : value == null ? '未记录' : String(value);
    function link(label, url) {
      const safe = publicURL(url);
      const element = node(safe ? 'a' : 'span', '', label);
      if (safe) { element.href = safe; element.target = '_blank'; element.rel = 'noopener noreferrer'; }
      return element;
    }
    function settings(value) {
      const wrapper = node('div', 'difference-configuration evaluation-card');
      wrapper.append(global.PrecisionEvaluationSettings.renderSettings(value, helpers));
      return wrapper;
    }
    function drawBar(value, domain, className = '') {
      const track = node('div', 'difference-track');
      track.setAttribute('aria-hidden', 'true');
      const geometry = global.PrecisionEvaluations.barGeometry(value, domain);
      if (!geometry) return track;
      const bar = node('span', `difference-bar ${className}`);
      bar.style.left = `${geometry.left}%`;
      bar.style.width = `${geometry.width}%`;
      const zero = node('span', 'difference-zero');
      zero.style.left = `${geometry.zero}%`;
      track.append(bar, zero);
      return track;
    }
    function metricPanel(metric) {
      const box = node('div', 'difference-metric');
      box.append(node('dt', '', metricLabel(metric)), node('dd', 'difference-metric-value', exactValue(metric.value)),
        node('dd', 'difference-metric-unit', unitLabel(metric.unit)));
      if (metric.numerator != null || metric.denominator != null) {
        box.append(node('dd', 'difference-metric-detail', `${exactValue(metric.numerator)} / ${exactValue(metric.denominator)}`));
      }
      const sides = list(metric.side_values);
      const paired = sides.length === 2 && sides.every((side) => finite(side.value)) && sides[0].value !== sides[1].value;
      const domain = paired ? [Math.min(0, ...sides.map((side) => side.value)), Math.max(0, ...sides.map((side) => side.value))] : null;
      sides.forEach((side, index) => {
        const row = node('dd', 'difference-side-value');
        row.append(node('span', '', side.label || `侧 ${index + 1}`), node('strong', '', exactValue(side.value)));
        if (paired) row.append(drawBar(side.value, domain, index ? 'difference-bar-peer' : ''));
        box.append(row);
      });
      if (paired) box.append(node('dd', 'difference-metric-detail', '两侧共用线性零基线；灰 / 绿表示比较侧，不表示优劣。'));
      if (metric.source_pointer) {
        const pointer = node('dd', 'difference-metric-detail', `来源 ${metric.source_pointer}`);
        box.append(pointer);
      }
      if (metric.coordinate) box.append(settings({ coordinate: metric.coordinate }));
      return box;
    }
    function card(record) {
      const item = node('details', 'repair-item repair-overview difference-card');
      item.id = `difference-${record.id}`;
      item.dataset.comparisonKind = record.comparison_kind;
      const summary = node('summary', 'issue-summary');
      const metadata = node('div', 'repair-metadata');
      metadata.append(node('span', 'repair-version', record.platform || '平台未记录'),
        node('span', `difference-kind difference-kind-${record.comparison_kind}`, kindLabel(record.comparison_kind)));
      summary.append(metadata, node('h4', 'overview-title', record.label || record.id));
      const headline = node('div', 'difference-headline-metrics');
      representativeMetrics(record).forEach((metric) => {
        const value = node('strong', '', compactNumber(metric.value));
        value.title = exactValue(metric.value);
        const block = node('div');
        block.append(node('span', '', metricLabel(metric)), value, node('small', '', unitLabel(metric.unit)));
        headline.append(block);
      });
      if (!headline.childElementCount) headline.append(node('p', '', '展开查看采集状态与诊断证据'));
      const scope = [record.stage, record.mode].filter(Boolean).join(' · ');
      summary.append(headline, node('p', 'overview-scope', scope || record.scope || '比较范围见详情'));
      const disclosure = node('div', 'overview-disclosure');
      disclosure.append(node('span', '', '指标、配置与证据'), node('span', '', `${list(record.metrics).length} 项指标`));
      summary.append(disclosure);
      const content = node('div', 'issue-content difference-card-content');
      let populated = false;
      function populate() {
        if (populated) return;
        populated = true;
        const toolbar = node('div', 'difference-record-toolbar');
        toolbar.append(node('span', '', record.status === 'completed' ? '采集与比较完成' : text(record.status, '状态未记录')));
        const permalink = node('a', '', '条目链接 ↗');
        permalink.href = `#difference-${encodeURIComponent(record.id)}`;
        toolbar.append(permalink);
        content.append(toolbar, node('p', 'difference-scope-note', record.scope), node('p', 'difference-kind-note', kindNotes[record.comparison_kind] || '请按原始证据中的范围解释。'));
        const metrics = node('dl', 'difference-metrics');
        list(record.metrics).forEach((metric) => metrics.append(metricPanel(metric)));
        content.append(metrics);
        const details = node('details', 'details difference-evidence');
        details.append(node('summary', '', '查看两侧配置、控制条件和来源'));
        const grid = node('div', 'detail-grid');
        const comparison = node('section', 'detail-section');
        comparison.append(node('h6', '', '01 / 比较条件'));
        comparison.append(settings(record.configuration || { common_conditions: record.common_conditions, differences: record.differences }));
        if (list(record.coverage).length) {
          comparison.append(node('h6', 'difference-side-title', '采集覆盖'), settings({ coverage: record.coverage }));
        }
        if (record.side_a || record.side_b) {
          [record.side_a, record.side_b].filter(Boolean).forEach((side) => {
            comparison.append(node('h6', 'difference-side-title', side.label || side.id || '比较侧'), settings(side));
          });
        }
        const provenance = node('section', 'detail-section');
        provenance.append(node('h6', '', '02 / 原始证据'));
        const sources = list(record.sources).length ? record.sources : record.source ? [record.source] : [];
        sources.forEach((source) => {
          const entry = node('div', 'difference-source');
          entry.append(link(source.path || source.label || '查看原始证据', source.url));
          if (source.archive_member) entry.append(node('p', 'difference-pointer', `ZIP 内文件：${source.archive_member}`));
          if (source.branch) entry.append(node('p', '', source.branch));
          if (source.commit) entry.append(node('code', '', source.commit));
          if (source.json_pointer) entry.append(node('p', 'difference-pointer', `JSON ${source.json_pointer}`));
          provenance.append(entry);
        });
        if (list(record.notes).length) {
          provenance.append(node('h6', 'difference-side-title', '范围与补充说明'));
          const notes = node('ul');
          record.notes.forEach((note) => notes.append(node('li', '', typeof note === 'string' ? note : JSON.stringify(note))));
          provenance.append(notes);
        }
        provenance.append(node('p', 'difference-acceptance', record.acceptance == null ? '数值验收：未记录结论。差异大小不等于任务效果变化。' : `数值验收：${typeof record.acceptance === 'object' ? JSON.stringify(record.acceptance) : record.acceptance}`));
        grid.append(comparison, provenance);
        details.append(grid);
        content.append(details);
      }
      // Populate on open to keep large, comprehensive evidence collections responsive.
      item.addEventListener('toggle', () => { if (item.open) populate(); });
      item.append(summary, content);
      return item;
    }
    function explorer(group, groupIndex) {
      const options = metricOptions(group.records);
      if (!options.length) return null;
      const element = node('details', 'difference-explorer');
      element.append(node('summary', '', '按指标比较各配置'));
      element.open = groupIndex === 0;
      let populated = false;
      element.addEventListener('toggle', () => {
        if (!element.open || populated) return;
        populated = true;
        const body = node('div', 'difference-explorer-body');
        const label = node('label', 'difference-explorer-control');
        label.append(node('span', '', '比较指标'));
        const select = node('select');
        select.setAttribute('aria-label', `${group.label} 比较指标`);
        options.forEach((option) => {
          const item = node('option', '', `${option.label} · ${unitLabel(option.unit)} · ${kindLabel(option.kind)} · ${option.stage || '汇总'} · ${option.source.split('/').pop() || '来源未记录'} · ${option.count} 条`);
          item.value = option.key;
          select.append(item);
        });
        label.append(select);
        const plot = node('div', 'difference-plot');
        const description = node('p', 'difference-chart-note', '同一比较类型、来源和阶段的同名同单位指标共用线性零基线；每行对应一条测量，不合并平均。条长不表示验收结果，微小差异以精确数值为准。');
        let chartPage = 0;
        const chartPageSize = 12;
        const chartNavigation = node('div', 'difference-pagination');
        function updatePlot() {
          const entries = group.records.flatMap((record) => list(record.metrics).filter((metric) => metricKey(metric, record) === select.value && finite(metric.value)).map((metric) => ({ record, metric })));
          const domain = [Math.min(0, ...entries.map(({ metric }) => metric.value)), Math.max(0, ...entries.map(({ metric }) => metric.value))];
          const pageCount = Math.max(1, Math.ceil(entries.length / chartPageSize));
          chartPage = Math.min(chartPage, pageCount - 1);
          const rows = entries.slice(chartPage * chartPageSize, (chartPage + 1) * chartPageSize).map(({ record, metric }) => {
            const row = node('div', 'difference-plot-row');
            const recordLink = node('a', 'difference-plot-label', record.label || record.id);
            recordLink.href = `#difference-${encodeURIComponent(record.id)}`;
            row.append(recordLink, node('span', 'difference-plot-kind', `${kindLabel(record.comparison_kind)} · ${record.platform || ''}`),
              node('strong', 'difference-chart-value', `${exactValue(metric.value)} ${unitLabel(metric.unit)}`), drawBar(metric.value, domain, record.comparison_kind === 'cross-backend' ? 'difference-bar-peer' : ''));
            return row;
          });
          const axis = node('div', 'difference-axis');
          axis.append(node('span', '', exactValue(domain[0])), node('span', '', exactValue(domain[1])));
          plot.replaceChildren(...rows, axis);
          chartNavigation.replaceChildren();
          const previous = node('button', '', '上一页'); previous.type = 'button'; previous.disabled = chartPage === 0;
          const next = node('button', '', '下一页'); next.type = 'button'; next.disabled = chartPage === pageCount - 1;
          previous.addEventListener('click', () => { chartPage -= 1; updatePlot(); });
          next.addEventListener('click', () => { chartPage += 1; updatePlot(); });
          chartNavigation.append(node('span', '', `${entries.length} 条测量 · 第 ${chartPage + 1} / ${pageCount} 页`), previous, next);
          chartNavigation.hidden = pageCount < 2;
        }
        select.addEventListener('change', () => { chartPage = 0; updatePlot(); });
        body.append(label, description, plot, chartNavigation);
        element.append(body);
        updatePlot();
      });
      return element;
    }
    function overview() {
      const element = node('section', 'difference-overview');
      const title = node('div', 'difference-overview-title');
      title.append(node('h3', '', '比较类型与证据覆盖'), node('p', '', '汇集配对比较与逐层、逐阶段测量视图；跨分支重复证据已去重。'));
      element.append(title);
      const distribution = node('div', 'difference-distribution');
      Object.entries(kindLabels).forEach(([kind, label]) => {
        const count = records.filter((record) => record.comparison_kind === kind).length;
        if (!count) return;
        const row = node('div', 'difference-distribution-row');
        const name = node('span', '', label);
        row.append(name, node('strong', '', `${count} 条`), drawBar(count, [0, records.length], kind === 'cross-backend' ? 'difference-bar-peer' : ''));
        distribution.append(row);
      });
      element.append(distribution, node('p', 'difference-overview-note', 'GPU ↔ NPU 是直接跨平台比较；其余类型是平台内部或参考实现对比。共享输入不代表两侧软件栈、精度和并行策略完全相同。条形仅表示测量记录数量，同一实验可包含多条视图。'));
      return element;
    }
    const elements = records.length ? [overview()] : [];
    groups(records).forEach((group, groupIndex) => {
      const element = node('section', 'library-group difference-model');
      const header = node('div', 'library-group-header');
      const title = node('div');
      title.append(node('p', 'eyebrow', 'MODEL COMPARISONS'), node('h3', '', group.label));
      header.append(title, node('span', 'group-count', `${group.records.length} 条测量 · ${new Set(group.records.map((record) => record.comparison_kind)).size} 类协议`));
      element.append(header);
      const chart = explorer(group, groupIndex);
      if (chart) element.append(chart);
      const cards = node('div', 'repair-list difference-list');
      const navigation = node('div', 'difference-pagination');
      const pageSize = 6;
      let currentPage = -1;
      function showPage(page) {
        if (page === currentPage) return;
        currentPage = page;
        const pageCount = Math.ceil(group.records.length / pageSize);
        cards.replaceChildren(...group.records.slice(page * pageSize, (page + 1) * pageSize).map(card));
        navigation.replaceChildren();
        const previous = node('button', '', '上一页'); previous.type = 'button'; previous.disabled = page === 0;
        const next = node('button', '', '下一页'); next.type = 'button'; next.disabled = page === pageCount - 1;
        previous.addEventListener('click', () => { showPage(currentPage - 1); header.scrollIntoView({ block: 'start' }); });
        next.addEventListener('click', () => { showPage(currentPage + 1); header.scrollIntoView({ block: 'start' }); });
        const position = node('span', '', `${page * pageSize + 1}–${Math.min((page + 1) * pageSize, group.records.length)} / ${group.records.length} 条`);
        const jump = node('select'); jump.setAttribute('aria-label', `${group.label} 记录页码`);
        for (let index = 0; index < pageCount; index += 1) {
          const option = node('option', '', `第 ${index + 1} / ${pageCount} 页`); option.value = index; jump.append(option);
        }
        jump.value = page;
        jump.addEventListener('change', () => { showPage(Number(jump.value)); header.scrollIntoView({ block: 'start' }); });
        navigation.append(position, previous, jump, next);
        navigation.hidden = pageCount < 2;
      }
      visibleGroups.set(group.label, { records: group.records, pageSize, showPage });
      showPage(0);
      element.append(cards, navigation);
      elements.push(element);
    });
    const sources = node('details', 'difference-source-index');
    const branches = list(archive.branches).length ? archive.branches : list(archive.generated_from);
    sources.append(node('summary', '', `数据来源与收集范围 · ${branches.length} 个分支 · ${list(archive.sources).length} 份证据`));
    let sourcesPopulated = false;
    sources.addEventListener('toggle', () => {
      if (!sources.open || sourcesPopulated) return;
      sourcesPopulated = true;
      const body = node('div', 'difference-source-index-body');
      if (archive.coverage) body.append(settings(archive.coverage));
      if (list(archive.notes).length) {
        const notes = node('ul', 'difference-branch-list');
        archive.notes.forEach((note) => notes.append(node('li', '', typeof note === 'string' ? note : JSON.stringify(note))));
        body.append(notes);
      }
      const branchList = node('ul', 'difference-branch-list');
      branches.forEach((branch) => {
        const item = node('li');
        const name = branch.name || branch.branch || branch.ref;
        item.append(node('strong', '', name), node('code', '', branch.commit || branch.sha || ''));
        branchList.append(item);
      });
      body.append(branchList);
      if (typeof archive.raw_results_url === 'string' && /^(?:\.\/)?(?:data|evidence)\/[a-zA-Z0-9_./-]+\.json$/.test(archive.raw_results_url) && !archive.raw_results_url.split('/').includes('..')) {
        const raw = node('a', 'difference-raw-link', '下载完整原始结果 JSON ↗');
        raw.href = archive.raw_results_url;
        raw.download = '';
        body.append(raw);
      }
      const index = node('div', 'difference-source-table');
      list(archive.sources).forEach((source) => {
        const entry = node('div', 'difference-source');
        entry.append(link(source.path || source.label || source.id, source.url));
        if (source.archive_member) entry.append(node('p', 'difference-pointer', `ZIP 内文件：${source.archive_member}`));
        if (source.sha256) entry.append(node('code', '', `SHA-256 ${source.sha256}`));
        const aliases = list(source.branch_aliases).length ? source.branch_aliases : sorted(list(source.origins).map((origin) => origin.branch));
        if (aliases.length) entry.append(node('p', '', `出现于 ${aliases.join(' · ')}`));
        index.append(entry);
      });
      body.append(index);
      sources.append(body);
    });
    if (records.length) elements.push(sources);
    return elements;
  }
  const api = { kindLabels, kindLabel, modelLabel, platformLabels, libraries, facets, filterRecords, groups,
    metricKey, metricOptions, metricLabel, compactNumber, representativeMetrics, revealRecord, renderArchive };
  if (typeof module === 'object' && module.exports) module.exports = api;
  global.PrecisionDifferences = api;
})(globalThis);
