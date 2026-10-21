/* Optional browser regression: run with Playwright and an installed Chromium. */
const { chromium } = require('playwright');
const { spawn } = require('node:child_process');
const { readFile } = require('node:fs/promises');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const data = require('../site/data/evaluations.json');
const repairs = require('../site/data/repairs.json');
(async () => {
  const server = spawn(process.env.PRECISION_PYTHON || 'python3', ['-u', '-m', 'http.server', '0', '--bind', '127.0.0.1', '--directory', path.join(root, 'site')], { stdio: ['ignore', 'pipe', 'pipe'] });
  let browser;
  try {
    const port = await new Promise((resolve, reject) => {
      let output = '';
      server.once('error', reject);
      server.once('exit', (code) => reject(new Error(`Preview server exited ${code}`)));
      server.stdout.on('data', (chunk) => {
        output += chunk.toString();
        const match = output.match(/port (\d+)/);
        if (match) resolve(Number(match[1]));
      });
      setTimeout(() => reject(new Error('Preview server unavailable')), 10000).unref();
    });
    const executablePath = process.env.PRECISION_BROWSER_EXECUTABLE;
    browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    const url = `http://127.0.0.1:${port}/`;
    const cards = page.locator('.evaluation-card');
    await page.goto(url, { waitUntil: 'networkidle' });
    await page.waitForSelector('.verification-card');
    assert.equal(await page.locator('.verification-card').count(), repairs.repairs.length);
    await page.getByRole('tab', { name: /精度评测历史/ }).click();
    await page.waitForSelector('.evaluation-card');
    assert.equal(await cards.count(), data.evaluations.length);
    assert.equal(await page.locator('#verification-count').innerText(), String(data.evaluations.reduce((n, r) => n + r.comparisons.length, 0)));
    assert.equal(await page.locator('#tab-evaluations').getAttribute('aria-selected'), 'true');
    assert.equal(await page.locator('#archive-panel').getAttribute('aria-labelledby'), 'tab-evaluations');
    await page.selectOption('#library', 'TRL');
    const trlRecords = data.evaluations.filter((r) => r.sides.some((s) => s.libraries.some((lib) => lib.name === 'TRL')));
    assert.equal(await cards.count(), trlRecords.length);
    await page.selectOption('#task', 'inference');
    assert.equal(await cards.count(), trlRecords.filter((r) => r.task === 'inference').length);
    await page.locator('#search').fill('precision-history-no-matching-fixture-739af7');
    assert.equal(await cards.count(), 0);
    assert.equal(await page.locator('#state').isVisible(), true);
    await page.locator('#reset').click();
    assert.equal(await cards.count(), data.evaluations.length);
    await page.selectOption('#hardware', 'Ascend 910C');
    assert.equal(await cards.count(), data.evaluations.filter((r) => r.sides.some((s) => s.hardware === 'Ascend 910C')).length);
    await page.locator('#reset').click();
    const restoration = data.evaluations.find((r) => r.id === 'trl-sfttrainer-checkpoint-20261021');
    const card = page.locator('#evaluation-' + restoration.id);
    for (const comparison of restoration.comparisons) {
      await card.locator('.comparison-picker select').selectOption(comparison.id);
      const values = await card.locator('.evaluation-value').evaluateAll((elements) => elements.map((element) => element.firstChild.textContent));
      assert.deepEqual(values, comparison.metrics.map((metric) => String(metric.value)));
      assert.equal(await card.locator('.comparison-title').innerText(), comparison.side_ids.map((id) => restoration.sides.find((s) => s.id === id).label).join(' ↔ '));
    }
    // Compare rendered charts with recorded values and independent linear geometry.
    let chartComparisons = 0, chartMetrics = 0, drawnCharts = 0;
    for (const record of data.evaluations) {
      const current = page.locator('#evaluation-' + record.id);
      for (const comparison of record.comparisons) {
        await current.locator('.comparison-picker select').selectOption(comparison.id);
        const rendered = await current.locator('.evaluation-metric').evaluateAll((items) => items.map((item) => {
          const chart = item.querySelector('.evaluation-chart');
          return { id: item.dataset.metricId, value: item.querySelector('.evaluation-value').firstChild.textContent,
            kind: chart.dataset.kind,
            domain: chart.hasAttribute('data-domain-min') ? [Number(chart.dataset.domainMin), Number(chart.dataset.domainMax)] : null,
            bars: [...chart.querySelectorAll('.evaluation-bar')].map((bar) => ({
              value: Number(bar.dataset.value), side: bar.dataset.sideId || null,
              left: parseFloat(bar.style.left), width: parseFloat(bar.style.width),
              zero: parseFloat(bar.parentElement.querySelector('.evaluation-zero').style.left) })),
            labels: [...chart.querySelectorAll('.evaluation-bar-label,.evaluation-pair-label')].map((label) => label.textContent),
            values: [...chart.querySelectorAll('.evaluation-bar-value,.evaluation-pair-value')].map((value) => value.textContent) };
        }));
        assert.equal(rendered.length, comparison.metrics.length);
        rendered.forEach((graphic, index) => {
          const metric = comparison.metrics[index];
          const context = record.id + '/' + comparison.id + '/' + metric.id;
          assert.equal(graphic.id, metric.id, context);
          assert.equal(graphic.value, String(metric.value), context);
          const expected = metric.side_values ? comparison.side_ids.map((id) => metric.side_values[id]) : [];
          const drawPair = expected.length === 2 && expected[0] !== expected[1];
          assert.equal(graphic.kind, drawPair ? 'sides' : 'none', context);
          if (metric.side_values) {
            assert.deepEqual(graphic.values, expected.map(String), context);
            assert.deepEqual(graphic.labels, comparison.side_ids.map((id) => record.sides.find((side) => side.id === id).label), context);
          }
          if (!drawPair) {
            assert.equal(graphic.bars.length, 0, context);
            assert.equal(graphic.domain, null, context);
            return;
          }
          drawnCharts++;
          assert.deepEqual(graphic.bars.map((bar) => bar.value), expected, context);
          assert.deepEqual(graphic.bars.map((bar) => bar.side), comparison.side_ids, context);
          assert.deepEqual(graphic.domain, [Math.min(0, ...expected), Math.max(0, ...expected)], context);
          const [minimum, maximum] = graphic.domain;
          const span = maximum - minimum;
          const zero = span ? -minimum / span * 100 : 0;
          graphic.bars.forEach((bar) => {
            const end = span ? (bar.value - minimum) / span * 100 : 0;
            assert.ok([bar.left, bar.width, bar.zero].every(Number.isFinite), context);
            assert.ok(Math.abs(bar.zero - zero) < 0.0001, context);
            assert.ok(Math.abs(bar.left - Math.min(zero, end)) < 0.0001, context);
            assert.ok(Math.abs(bar.width - Math.abs(end - zero)) < 0.0001, context);
            if (bar.value === 0) assert.equal(bar.width, 0, context);
          });
        });
        const coverage = await current.locator('.evaluation-coverage-heading strong').allTextContents();
        assert.deepEqual(coverage, comparison.coverage.map((row) => `${row.observed} / ${row.expected} ${row.unit}`));
        assert.equal(await current.locator('.evaluation-coverage-track').count(), 0);
        chartComparisons++;
        chartMetrics += rendered.length;
      }
    }
    // Reproduce the reported long JSON/forward-shape display and verify literal text safety.
    const readingRecord = data.evaluations.find((record) => record.id === 'llamacpp-local-qwen2-5-0-5b-instruct-20261019');
    const reading = page.locator('#evaluation-' + readingRecord.id);
    await reading.locator('.details > summary').click();
    const firstConfiguration = reading.locator('.evaluation-configuration-side').first();
    assert.equal(await firstConfiguration.evaluate((element) => element.open), true);
    const settings = firstConfiguration.locator('.evaluation-settings');
    const readableText = await settings.textContent();
    assert.equal(readableText.includes('recorded_conditions'), false);
    assert.equal(readableText.includes('"model":'), false);
    assert.equal(readableText.includes('one6-token'), false);
    assert.ok(readableText.includes('1 次 6 token 预填充，随后 2 次单 token 缓存解码'));
    assert.ok(readableText.includes(readingRecord.model.name));
    const prompt = readingRecord.sides[0].settings.recorded_conditions.inputs.prompt;
    assert.ok((await settings.locator('.evaluation-setting-row[data-setting-key="prompt"]').innerText()).includes('[' + prompt.join(', ') + ']'));
    const revision = settings.locator('.evaluation-setting-row[data-setting-key="revision"] details').first();
    assert.equal(await revision.evaluate((element) => element.open), false);
    const literalSafety = await page.evaluate(() => {
      const node = (tag, className, text) => {
        const element = document.createElement(tag); element.className = className || '';
        if (text !== undefined) element.textContent = String(text);
        return element;
      };
      const hostile = '<img src=x onerror="window.precisionUnsafeMarkup=true">';
      const config = {recorded_conditions: {model: {name: hostile},
        inputs: {prompt: [1, 2, 3], long_prompt: Array.from({length: 30}, (_, i) => i)},
        numerics: {zero: 0, flag: false, missing: null, empty: ''}}};
      const before = JSON.stringify(config);
      const host = PrecisionEvaluationSettings.renderSettings(config, {node});
      document.body.append(host);
      const values = {};
      host.querySelectorAll('.evaluation-setting-row').forEach((row) => {
        values[row.dataset.settingKey] = row.querySelector('dd').textContent;
      });
      const result = {literal: host.textContent.includes(hostile), images: host.querySelectorAll('img').length,
        executed: Boolean(window.precisionUnsafeMarkup), unchanged: before === JSON.stringify(config), values,
        longArrayCollapsed: host.querySelector('[data-setting-key="long_prompt"] details').open === false};
      host.remove();
      return result;
    });
    assert.equal(literalSafety.literal, true);
    assert.equal(literalSafety.images, 0);
    assert.equal(literalSafety.executed, false);
    assert.equal(literalSafety.unchanged, true);
    assert.equal(literalSafety.longArrayCollapsed, true);
    assert.equal(literalSafety.values.zero, '0');
    assert.equal(new Set(['zero', 'flag', 'missing', 'empty'].map((key) => literalSafety.values[key])).size, 4);
    if (process.env.PRECISION_CONFIG_SCREENSHOT) {
      await firstConfiguration.screenshot({path: process.env.PRECISION_CONFIG_SCREENSHOT});
    }
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#download').click();
    const download = await downloadEvent;
    assert.equal(download.suggestedFilename(), 'precision-skills-evaluations.json');
    assert.deepEqual(JSON.parse(await readFile(await download.path(), 'utf8')), data);
    await page.goto(url + '#evaluation-' + restoration.id, { waitUntil: 'networkidle' });
    await page.waitForSelector('.evaluation-card');
    assert.equal(await page.locator('#evaluation-' + restoration.id + ' .details').evaluate((e) => e.open), true);
    await page.locator('#tab-repairs').click();
    await page.waitForSelector('.verification-card');
    await page.goBack({ waitUntil: 'networkidle' });
    await page.waitForSelector('.evaluation-card');
    assert.equal(await page.locator('#tab-evaluations').getAttribute('aria-selected'), 'true');
    await page.locator('#tab-evaluations').focus();
    await page.keyboard.press('Home');
    await page.waitForSelector('.verification-card');
    await page.keyboard.press('End');
    await page.waitForSelector('.evaluation-card');
    await page.locator('#tab-repairs').click();
    await page.waitForSelector('.verification-card');
    await page.route('**/data/evaluations.json', (route) => route.fulfill({ status: 503, body: 'unavailable' }));
    await page.locator('#tab-evaluations').click();
    await page.getByRole('button', { name: '重新加载' }).waitFor();
    assert.equal(await page.locator('#download').isDisabled(), true);
    await page.unroute('**/data/evaluations.json');
    await page.getByRole('button', { name: '重新加载' }).click();
    await page.waitForSelector('.evaluation-card');
    if (process.env.PRECISION_CHART_SCREENSHOT) {
      const illustration = page.locator('#evaluation-tensorrt-llm-qwen25-05b-20261020');
      await illustration.locator('.comparison-picker select').selectOption('targets');
      await illustration.screenshot({ path: process.env.PRECISION_CHART_SCREENSHOT });
    }
    await page.screenshot({ path: process.env.PRECISION_DESKTOP_SCREENSHOT || '/tmp/precision-evaluations-desktop.png' });
    for (const width of [375, 768]) {
      await page.setViewportSize({ width, height: 950 });
      await page.locator('#evaluation-' + readingRecord.id + ' .details').first().evaluate((element) => { element.open = true; });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `Overflow at ${width}px`);
    }
    await page.setViewportSize({ width: 375, height: 950 });
    await page.screenshot({ path: process.env.PRECISION_MOBILE_SCREENSHOT || '/tmp/precision-evaluations-mobile.png' });
    await page.locator('#evaluation-' + restoration.id).screenshot({ path: '/tmp/precision-evaluations-mobile-card.png' });
    await page.goto(url + '#repair-' + repairs.repairs[0].id, { waitUntil: 'networkidle' });
    await page.waitForSelector('.verification-card');
    assert.equal(await page.locator('#repair-' + repairs.repairs[0].id + ' details').evaluate((e) => e.open), true);
    // A response from an abandoned tab must not replace the current archive.
    await page.route('**/data/evaluations.json', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 200));
      await route.fulfill({ json: data }).catch(() => {});
    });
    await page.locator('#tab-evaluations').click();
    await page.locator('#tab-repairs').click();
    await page.waitForSelector('.verification-card');
    await page.waitForTimeout(300);
    assert.equal(await cards.count(), 0);
    assert.equal(await page.locator('.verification-card').count(), repairs.repairs.length);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ status: 'passed', evaluations: data.evaluations.length, repairRecords: repairs.repairs.length,
      chartComparisons, chartMetrics, drawnCharts,
      checks: ['tabs', 'task/library/hardware filters', 'all comparison values', 'zero and tiny values', 'JSON download',
        'permalinks', 'back navigation', 'keyboard', 'load failure/retry', 'abandoned requests', 'mobile layout'] }));
  } finally {
    if (browser) await browser.close();
    server.kill('SIGTERM');
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
