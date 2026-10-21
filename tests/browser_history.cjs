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
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#download').click();
    const download = await downloadEvent;
    assert.equal(download.suggestedFilename(), 'precision-skills-evaluations.json');
    assert.deepEqual(JSON.parse(await readFile(await download.path(), 'utf8')), data);
    await page.goto(url + '#evaluation-' + restoration.id, { waitUntil: 'networkidle' });
    await page.waitForSelector('.evaluation-card');
    assert.equal(await page.locator('#evaluation-' + restoration.id + ' details').evaluate((e) => e.open), true);
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
    await page.screenshot({ path: process.env.PRECISION_DESKTOP_SCREENSHOT || '/tmp/precision-evaluations-desktop.png', fullPage: true });
    for (const width of [375, 768]) {
      await page.setViewportSize({ width, height: 950 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `Overflow at ${width}px`);
    }
    await page.setViewportSize({ width: 375, height: 950 });
    await page.screenshot({ path: process.env.PRECISION_MOBILE_SCREENSHOT || '/tmp/precision-evaluations-mobile.png', fullPage: true });
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
      checks: ['tabs', 'task/library/hardware filters', 'all comparison values', 'zero and tiny values', 'JSON download',
        'permalinks', 'back navigation', 'keyboard', 'load failure/retry', 'abandoned requests', 'mobile layout'] }));
  } finally {
    if (browser) await browser.close();
    server.kill('SIGTERM');
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
