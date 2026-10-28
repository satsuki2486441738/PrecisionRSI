'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const labels = {running:'运行中',completed:'已结束',failed:'执行失败',accepted:'已接受',rejected:'未接受',evaluating:'评测中',construction:'构建候选',evaluation:'独立评测',finished:'已结束'};
  const kinds = {tool:'工具',context:'上下文',procedure:'执行流程',data:'数据',weights:'权重'};
  const titles = {continuation:'反馈驱动 · 继续迭代',diagnostics:'原生错误反馈',interface:'接口与工具复用',budget:'预算与工具接口',recovery:'执行恢复',autonomous:'自主瓶颈选择',hour:'首轮自主改进'};
  let snapshot, selected, request = 0;
  const text = (tag, value, cls) => {const el=document.createElement(tag);el.textContent=value;if(cls)el.className=cls;return el;};
  const num = n => typeof n==='number' && Number.isFinite(n) ? n.toLocaleString('zh-CN',{maximumFractionDigits:0}) : '未知';
  const seconds = n => typeof n==='number' ? `${num(n)} s` : '未知';
  const version = v => !v || v==='empty' ? 'empty' : v.slice(0,10);
  function date(id) {const match=id.match(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/);if(!match)return '时间未知';return new Date(`${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:${match[6]}Z`).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});}
  function title(run) {return titles[run.id.replace(/^.*?-rsi-/,'').replace(/-\d{8}T.*$/,'')] || '迭代试跑';}
  function badge(status) {return text('span',labels[status] || status,'badge '+status);}
  function sumTokens(run) {const vals=Object.values(run.phases).map(p=>p.usage.total_tokens);const known=vals.filter(v=>typeof v==='number');return known.length?{value:known.reduce((a,b)=>a+b,0),partial:known.length!==vals.length}:null;}
  function stats() {
    const current=snapshot.runs.find(r=>r.id===snapshot.current_run);
    const el=$('overview');el.replaceChildren();
    const data=[['当前运行',labels[current?.status] || '未知',current?`${labels[current.phase] || current.phase} · ${seconds(current.elapsed_seconds)}`:'尚无运行'],['当前启用版本',version(current?.active_version),'仅在当前运行的版本库内生效'],['本轮候选接受',`${num(current?.accepted)} / ${current?.rounds.length || 0}`,'接受数 / 已记录轮次'],['当前已观测 token',current&&sumTokens(current)?(sumTokens(current).partial?'≥ ':'')+num(sumTokens(current).value):'未知','含构建与评测；复用基线不重复计费']];
    data.forEach(([label,value,foot],i)=>{const div=text('div','','stat');div.append(text('span',label,'stat-label'),text('strong',value,'stat-value'+(i===1?' version':'')),text('span',foot,'stat-foot'));el.append(div);});
  }
  function runList() {
    const container=$('runs');container.replaceChildren();const q=$('run-search').value.trim().toLowerCase();
    snapshot.runs.filter(r=>r.id.toLowerCase().includes(q)||title(r).includes(q)).forEach(run=>{
      const button=text('button','','run-button'+(run.id===selected?' active':''));button.type='button';button.setAttribute('aria-pressed',String(run.id===selected));
      const row=text('span','','run-date');row.append(text('span',date(run.id)),badge(run.status));button.append(text('span',title(run),'run-name'),row);
      button.onclick=()=>{selected=run.id;history.replaceState(null,'','#'+encodeURIComponent(selected));$('round-filter').value='all';render();};container.append(button);
    });$('run-count').textContent=snapshot.runs.length;
  }
  function renderRounds(run) {
    const holder=$('rounds');holder.replaceChildren();
    const filter=$('round-filter').value;
    const rounds=run.rounds.filter(r=>filter==='all'||r.status===filter);
    if(run.status==='running'&&filter==='all') {
      const live=text('div','','empty');live.append(text('strong',`正在${labels[run.phase] || '迭代'}…`));
      const calls=run.live_activity;live.append(text('p',calls?`已观测 ${num(calls.tool_calls)} 次工具调用。完整轮次完成后显示变更与判定。`:'等待下一份完整轮次回执。'));
      if(calls&&Object.keys(calls.tools).length)live.append(text('span',Object.entries(calls.tools).map(([k,v])=>`${k} × ${v}`).join(' · '),'mono'));holder.append(live);
    }
    if(!rounds.length) {if(run.status!=='running'||filter!=='all')holder.append(text('p','没有符合条件的轮次。','empty'));return;}
    rounds.forEach(round=>{
      const details=document.createElement('details');details.className='round';
      const summary=document.createElement('summary');summary.append(text('span',String(round.index).padStart(2,'0'),'round-index'),text('span',round.changes.length?round.changes.map(c=>kinds[c.kind]||c.kind).join(' + '):'未形成可评测候选','round-heading'),badge(round.status));details.append(summary);
      const body=text('div','','round-body');
      if(round.version)body.append(text('p',`候选版本 ${version(round.version)}`,'mono muted'));
      round.changes.forEach(change=>{const item=text('div','','change');item.append(text('strong',kinds[change.kind]||change.kind),text('p',change.description||'已记录候选组件'));if(change.artifacts.length)item.append(text('p',change.artifacts.join(' · '),'mono muted'));body.append(item);});
      if(round.correct!==null&&round.correct!==undefined)body.append(text('p',round.correct?'独立任务正确完成。':'独立任务未通过。'));
      round.reasons.forEach(reason=>body.append(text('p',reason,'reason')));
      const costs=text('div','','round-cost');costs.append(text('span',`构建 ${seconds(round.construction.seconds)} / ${num(round.construction.usage.total_tokens)} tok`),text('span',`评测 ${seconds(round.evaluation.seconds)} / ${num(round.evaluation.usage.total_tokens)} tok`));body.append(costs);
      if(round.candidate_tokens!==null)body.append(text('p',`计入构建后的候选成本：${seconds(round.candidate_seconds)} · ${num(round.candidate_tokens)} tok。`,'muted'));
      if(round.activity)body.append(text('p',`已归档工具调用 ${num(round.activity.tool_calls)} 次 · ${Object.entries(round.activity.tools).map(([k,v])=>`${k} × ${v}`).join(' / ')}`,'muted'));
      details.append(body);holder.append(details);
    });
  }
  function render() {
    stats();runList();const run=snapshot.runs.find(r=>r.id===selected);if(!run)return;
    $('run-title').textContent=title(run);$('run-id').textContent=run.id;$('run-status').replaceChildren(badge(run.status));
    const meta=$('run-meta');meta.replaceChildren();[run.model,run.commit?`代码 ${run.commit.slice(0,8)}`:'代码版本未归档',`启用 ${version(run.active_version)}`].forEach(v=>meta.append(text('span',v)));
    $('scope').textContent=run.scope;
    const table=document.createElement('table');const head=document.createElement('thead');const row=document.createElement('tr');['阶段','时间','输入 token','输出 token','原始 token'].forEach(v=>row.append(text('th',v)));head.append(row);table.append(head);const body=document.createElement('tbody');
    Object.entries(run.phases).forEach(([key,phase])=>{const tr=document.createElement('tr');const name={baseline:'基线',construction:'构建',evaluation:'评测'}[key];[name+(key==='baseline'&&run.baseline.reused?' · 复用':''),seconds(phase.seconds),num(phase.usage.input_tokens),num(phase.usage.output_tokens),num(phase.usage.total_tokens)].forEach(v=>tr.append(text('td',v)));body.append(tr);});table.append(body);$('costs').replaceChildren(table);
    $('baseline-note').textContent=run.baseline.reused?`复用成功基线：历史成本 ${seconds(run.baseline.historical.seconds)} / ${num(run.baseline.historical.usage.total_tokens)} tok，本轮基线新增成本为 0。`:'时间与原始 token 分开记录。缓存拆分未取得时保持未知；本页不估算费用。';
    renderRounds(run);const audit=$('audit');audit.replaceChildren();if(run.stop_reason)audit.append(text('p',`停止原因：${run.stop_reason}`));run.maintenance_notes.forEach(v=>audit.append(text('p',v)));if(run.acceptance_note)audit.append(text('p',run.acceptance_note));if(run.evidence.archive_sha256)audit.append(text('p',`证据归档 SHA256 · ${run.evidence.archive_sha256}`,'mono'));
  }
  async function load() {
    const id=++request;$('refresh').disabled=true;
    try {const response=await fetch('./data/rsi.json',{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);const data=await response.json();if(id!==request)return;if(data.schema_version!==1||!Array.isArray(data.runs))throw new Error('快照格式不兼容');snapshot=data;const hash=decodeURIComponent(location.hash.slice(1));selected=data.runs.some(r=>r.id===selected)?selected:data.runs.some(r=>r.id===hash)?hash:data.current_run;
      const observed=data.observed_at?new Date(data.observed_at):null;const age=observed?(Date.now()-observed.getTime())/60000:null;
      $('freshness').textContent=age===null?'观测时间未知':age>90?'快照已超过 90 分钟':'观测快照 · 页面每分钟检查更新';
      $('snapshot-time').textContent=observed?observed.toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false})+' 北京时间':'—';$('message').textContent='';render();
    } catch(error) {$('message').textContent='快照读取失败，保留已有画面。'+error.message;}finally{if(id===request)$('refresh').disabled=false;}
  }
  $('refresh').onclick=load;$('run-search').oninput=()=>snapshot&&runList();$('round-filter').onchange=()=>snapshot&&renderRounds(snapshot.runs.find(r=>r.id===selected));window.addEventListener('hashchange',()=>{if(snapshot){const id=decodeURIComponent(location.hash.slice(1));if(snapshot.runs.some(r=>r.id===id)){selected=id;render();}}});
  load();setInterval(()=>{if(!document.hidden)load();},60000);
})();
