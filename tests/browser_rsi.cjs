/* Optional dashboard regression: Playwright plus existing Chromium. */
const {chromium}=require('playwright');
const {spawn}=require('node:child_process');
const {readFile}=require('node:fs/promises');
const path=require('node:path');
const assert=require('node:assert/strict');
(async()=>{
  const root=path.resolve(__dirname,'..');
  const data=JSON.parse(await readFile(path.join(root,'site/data/rsi.json'),'utf8'));
  const server=spawn(process.env.PRECISION_PYTHON||'python3',['-u','-m','http.server','0','--bind','127.0.0.1','--directory',path.join(root,'site')],{stdio:['ignore','pipe','pipe']});
  let browser;
  try {
    const port=await new Promise((resolve,reject)=>{let out='';server.once('error',reject);server.stdout.on('data',chunk=>{out+=chunk;const m=out.match(/port (\d+)/);if(m)resolve(m[1]);});setTimeout(()=>reject(Error('No preview server')),10000).unref();});
    browser=await chromium.launch({headless:true,...(process.env.PRECISION_BROWSER_EXECUTABLE?{executablePath:process.env.PRECISION_BROWSER_EXECUTABLE}:{})});
    const page=await browser.newPage({viewport:{width:1440,height:1050}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(`http://127.0.0.1:${port}/rsi.html`);
    await page.waitForFunction(()=>document.querySelectorAll('.run-button').length>0);
    assert.equal(await page.locator('.run-button').count(),data.runs.length);
    assert.equal(await page.locator('#run-id').textContent(),data.current_run);
    const current=data.runs.find(r=>r.id===data.current_run);
    if(current.baseline.reused)assert.match(await page.locator('#baseline-note').textContent(),/本轮基线新增成本为 0/);
    const old=data.runs.find(r=>r.rounds.length>0);
    await page.goto(`http://127.0.0.1:${port}/rsi.html#${old.id}`);
    await page.waitForFunction(id=>document.querySelector('#run-id').textContent===id,old.id);
    assert.equal(await page.locator('details.round').count(),old.rounds.length);
    await page.locator('details.round summary').first().click();
    assert.equal(await page.locator('details.round').first().getAttribute('open'),'');
    await page.selectOption('#round-filter','accepted');
    assert.equal(await page.locator('details.round').count(),old.rounds.filter(r=>r.status==='accepted').length);
    await page.selectOption('#round-filter','all');
    await page.fill('#run-search','nonexistent-run');assert.equal(await page.locator('.run-button').count(),0);
    await page.fill('#run-search','');assert.equal(await page.locator('.run-button').count(),data.runs.length);
    if(process.env.PRECISION_RSI_SCREENSHOT)await page.screenshot({path:process.env.PRECISION_RSI_SCREENSHOT,fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'mobile overflow');
    if(process.env.PRECISION_RSI_SCREENSHOT)await page.screenshot({path:process.env.PRECISION_RSI_SCREENSHOT.replace('.png','-mobile.png'),fullPage:true});
    await page.route('**/data/rsi.json',route=>route.abort());await page.click('#refresh');
    await page.waitForFunction(()=>document.querySelector('#message').textContent.includes('读取失败'));
    assert.equal(await page.locator('#run-id').textContent(),old.id);
    assert.deepEqual(errors,[]);
    console.log('RSI browser checks passed: selection, deep link, disclosures, filter, baseline costs, mobile, refresh failure.');
  } finally {if(browser)await browser.close();server.kill();}
})().catch(e=>{console.error(e);process.exitCode=1;});
