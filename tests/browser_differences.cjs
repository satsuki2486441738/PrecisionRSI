/* Optional UI regression: NODE_PATH must provide Playwright; uses an installed Chromium. */
const { chromium } = require('playwright');
const { spawn } = require('node:child_process');
const { readFile } = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const data = require('../site/data/differences.json');
const model = require('../site/differences.js');
(async () => {
  const server = spawn(process.env.PRECISION_PYTHON || 'python3', ['-u', '-m', 'http.server', '0', '--bind', '127.0.0.1', '--directory', path.join(root, 'site')], { stdio: ['ignore', 'pipe', 'pipe'] });
  let browser;
  try {
    const port = await new Promise((resolve, reject) => {
      server.once('error', reject);
      server.stdout.on('data', chunk => { const match = chunk.toString().match(/port (\d+)/); if (match) resolve(match[1]); });
      setTimeout(() => reject(new Error('Preview unavailable')), 10000).unref();
    });
    browser = await chromium.launch({ headless: true, ...(process.env.PRECISION_BROWSER_EXECUTABLE ? { executablePath: process.env.PRECISION_BROWSER_EXECUTABLE } : {}) });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const url = `http://127.0.0.1:${port}/archive.html`;
    await page.goto(url + '#differences');
    await page.waitForSelector('.difference-card');
    assert.equal(await page.locator('#verification-count').innerText(), String(data.records.length));
    assert.equal(await page.locator('.difference-model').count(), model.groups(data.records).length);
    assert.ok(await page.locator('.difference-card').count() <= 6 * data.models.length);
    assert.equal(await page.locator('#task-filter').isVisible(), false);

    await page.locator('#comparison-kind').selectOption('native-reference');
    await page.locator('#extra-filters > summary').click();
    const selectedModel = data.records.find(record => record.comparison_kind === 'native-reference').model;
    await page.locator('#model').selectOption(selectedModel);
    const expected = model.filterRecords(data.records, { model: selectedModel, comparison_kind: 'native-reference' });
    assert.ok((await page.locator('#result-count').innerText()).includes(`${expected.length} / ${data.records.length}`));
    for (const card of await page.locator('.difference-card').all()) assert.equal(await card.getAttribute('data-comparison-kind'), 'native-reference');
    const group = page.locator('.difference-model');
    const jump = group.locator(':scope > .difference-pagination select');
    if (expected.length > 6) {
      await jump.selectOption('1');
      assert.equal(await group.locator('.difference-card').first().getAttribute('id'), 'difference-' + model.groups(expected)[0].records[6].id);
    }
    await page.locator('#search').fill('no-such-survey-result-xyz');
    assert.equal(await page.locator('.difference-card').count(), 0);
    assert.equal(await page.locator('#state').isVisible(), true);
    await page.locator('#reset').click();

    // A deep link must reveal a record outside page one, then preserve source values.
    const target = [...data.records].reverse().find(record => record.metrics.length && record.comparison_kind === 'native-reference');
    await page.goto(url + '#difference-' + target.id);
    const card = page.locator('#difference-' + target.id);
    await card.locator('.difference-metric-value').first().waitFor();
    assert.equal(await card.evaluate(element => element.open), true);
    const values = await card.locator('.difference-metric-value').allTextContents();
    assert.deepEqual(values, target.metrics.map(metric => String(metric.value)));
    await card.locator('.difference-evidence > summary').click();
    const sourceLink = card.locator('.difference-source a').first();
    assert.ok((await sourceLink.getAttribute('href')).includes(target.source.commit));
    assert.ok((await card.locator('.difference-kind-note').innerText()).includes('不是 GPU 与 NPU'));

    await page.locator('#tab-differences').click();
    if (await page.locator('#reset').isVisible()) await page.locator('#reset').click();
    const downloadWait = page.waitForEvent('download');
    await page.locator('#download').click();
    const download = await downloadWait;
    assert.equal(download.suggestedFilename(), 'PrecisionRSI-differences.json');
    const downloaded = JSON.parse(await readFile(await download.path(), 'utf8'));
    assert.equal(downloaded.records.length, data.records.length);
    assert.deepEqual(downloaded.branches, data.branches);

    await page.locator('#tab-repairs').click();
    await page.waitForSelector('.repair-overview');
    await page.route('**/data/differences.json', route => route.fulfill({ status: 503, body: 'unavailable' }));
    await page.locator('#tab-differences').click();
    await page.getByRole('button', { name: '重新加载' }).waitFor();
    assert.equal(await page.locator('#download').isDisabled(), true);
    await page.unroute('**/data/differences.json');
    await page.getByRole('button', { name: '重新加载' }).click();
    await page.waitForSelector('.difference-card');

    await page.evaluate(() => window.scrollTo(0, 0));
    if (process.env.PRECISION_DIFFERENCES_DESKTOP_SCREENSHOT) await page.screenshot({ path: process.env.PRECISION_DIFFERENCES_DESKTOP_SCREENSHOT });
    for (const width of [375, 320]) {
      await page.setViewportSize({ width, height: 900 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `overflow at ${width}px`);
      await page.locator('.difference-card').first().locator(':scope > summary').click();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `expanded overflow at ${width}px`);
      await page.locator('.difference-card').first().locator(':scope > summary').click();
      await page.evaluate(() => window.scrollTo(0, 0));
      if (width === 375 && process.env.PRECISION_DIFFERENCES_MOBILE_SCREENSHOT) await page.screenshot({ path: process.env.PRECISION_DIFFERENCES_MOBILE_SCREENSHOT });
    }
    assert.deepEqual(errors, []);
    console.log(`Differences browser checks passed: ${data.records.length} views, filters, pagination, deep links, exact metrics, download, retry and mobile layout.`);
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
