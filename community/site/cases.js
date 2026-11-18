/* Precision case library, built from explicit record relations. */
(function (global) {
  'use strict';
  const {hardwarePlatforms}=typeof module!=='undefined'&&module.exports?require('./board-data.js'):global.NumericBridgeHardware;
  const Search=typeof module!=='undefined'&&module.exports?require('./search.js'):global.NumericBridgeSearch;
  const key = (kind, id) => kind + ':' + id;
  const choiceKey = value => String(value).trim().toLowerCase().replace(/\s+/g, ' ');
  const choiceGroups = {
    hardware:[['nvidia','NVIDIA',/\bnvidia\b|\brtx\b/i],['ascend','Ascend',/\bascend/i],
      ['amd','AMD',/\bamd\b|\bradeon\b/i],['intel','Intel',/\bintel\b|\bgaudi\b/i],['apple','Apple',/\bapple\b/i],['cpu','CPU',/\bcpu\b/i]],
    model:[['qwen','Qwen',/qwen/i],['deepseek','DeepSeek',/deepseek/i],['gemma','Gemma',/gemma/i],
      ['meta','Llama',/llama/i],['huggingface','SmolLM',/smollm/i],['openai','GPT',/\bgpt/i],['nvidia','Nemotron',/nemotron/i]]
  };
  const choiceGroup = (field, value) => choiceGroups[field]?.find(([, , pattern]) => pattern.test(value));
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const releaseVersion=/^v?\d+(?:\.\d+)+(?:[-.]?(?:a|b|rc|dev|post)\d*)*(?:\+(?:cpu|cu\d+|rocm[\d.]+))?$/i;
  const commitVersion=/^[a-f0-9]{7,64}$/i;
  // Parse display labels only. Filtering still uses the exact stored version.
  function versionParts({version,commit}) {
    const raw=String(version??'').trim();
    if(releaseVersion.test(raw))return {number:raw.replace(/^v/i,''),kind:'release'};
    const source=raw.match(/^(.+?)(?:\s*\/\s*source\s*@\s*|\s*@\s*|\s+from source archive\s+|\+g)[a-f0-9]{7,64}$/i);
    const number=source?.[1].replace(/^ascend-v/i,'v');
    if(number&&releaseVersion.test(number))return {number:/^ascend-/i.test(raw)?'Ascend '+number.slice(1):number.replace(/^v/i,''),kind:'source'};
    const build=raw.match(/^(.+)\+empty$/i)?.[1];
    if(build&&releaseVersion.test(build))return {number:build.replace(/^v/i,''),kind:'build'};
    const annotated=raw.match(/^(.+?)\s+\((?:source directory identity|main, .+)\)(?:\s+\[GCC [\d.]+\])?$/i)?.[1];
    if(annotated&&releaseVersion.test(annotated))return {number:annotated,kind:/source directory/i.test(raw)?'source':'release'};
    return {number:'',kind:source||commitVersion.test(raw)||commitVersion.test(commit??'')||/^(?:\S+ )?source(?:\b|@)/i.test(raw)?'source':''};
  }
  function buildIndex(relations, evaluations, repairs, searchMetadata=null) {
    const records = new Map(), members = new Map();
    for (const [kind, rows] of [['comparison', evaluations.evaluations], ['repair', repairs.repairs], ['entity', relations.entities]]) {
      for (const row of rows) records.set(key(kind === 'entity' ? row.kind : kind, row.id), row);
    }
    for (const link of relations.links) if (link.relation === 'includes') {
      const list = members.get(link.from.id) || []; list.push(link.to); members.set(link.from.id, list);
    }
    const cases = relations.entities.filter(r => r.kind === 'case').map(row => {
      const entries = members.get(row.id) || [];
      const observations = entries.filter(r => ['comparison', 'repair'].includes(r.kind)).map(r => ({kind:r.kind, record:records.get(key(r.kind,r.id))}));
      const category = row.details.category || (entries.some(r => r.kind === 'repair') ? 'repair-investigation' : 'precision-evaluation');
      const configurations = observations.flatMap(({kind,record:r}) => (kind === 'comparison' ? r.sides : [r.environment]).map(side => ({
        hardware:side.hardware, platforms:hardwarePlatforms(side.hardware), model:r.model?.name, libraries:side.libraries || [], settings:side.settings || {}, sideId:side.id, recordId:r.id, kind,
        tasks:r.tasks?.length ? r.tasks : (r.task ? [r.task] : (row.details.request.task ? [row.details.request.task] : []))
      })));
      const count = kind => entries.filter(e => e.kind === kind).length;
      const search = [row.id,row.details.title,row.scope,...Object.values(row.details.request),...entries.map(e => records.get(key(e.kind,e.id))?.details?.title),
        ...configurations.flatMap(c => [c.hardware,...c.platforms,c.model,...c.libraries.flatMap(l => [l.name,l.version])])].join(' ').toLowerCase();
      return {row, category, entries, observations, configurations, search,
        counts:{issue:count('issue'),candidate:count('candidate'),verification:count('verification'),measurement:count('measurement')}};
    });
    if(searchMetadata){
      const configs=new Map(searchMetadata.cases.flatMap(c=>c.configurations).map(c=>[c.id,c]));
      for(const c of cases)for(const config of c.configurations){
        config.searchConfig=configs.get(`${config.kind}:${config.recordId}:${config.sideId??0}`);
        if(config.searchConfig)config.platforms=config.searchConfig.platforms;
        config.searchContext={aliases:searchMetadata.aliases,hardware_rules:searchMetadata.hardware_rules};
      }
    }
    return {relations,records,cases,search:searchMetadata,content:searchMetadata?.content||{}};
  }
  function retrieve(index,filters,options={}) {
    return Search.search(index.search,Object.fromEntries([...configurationFields.map(k=>[k,filters[k]||'']),['query',filters.q||'']]),options);
  }
  function filterCases(index, filters) {
    if (filters.version && !filters.library) return [];
    return index.cases.filter(c => (!filters.category || c.category === filters.category) &&
      (!filters.q || c.search.includes(filters.q.trim().toLowerCase())) &&
      c.configurations.some(config => configurationMatches(config,filters)));
  }
  const configurationFields=['hardware','model','model_size','library','version','task'];
  function configurationDifferences(config,filters) {
    if(config.searchConfig&&filters.hardwareMode==='platform'){
      const request=Search.normalizeRequest(Object.fromEntries(configurationFields.map(k=>[k,filters[k]||''])),config.searchContext);
      return Search.configurationDifferences(config.searchConfig,request);
    }
    return configurationFields.filter(field=>filters[field] && !(
      field==='library' ? config.libraries.some(lib=>lib.name===filters.library) :
      field==='version' ? filters.library && config.libraries.some(lib=>lib.name===filters.library&&lib.version===filters.version) :
      field==='task' ? config.tasks.includes(filters.task) :
      field==='hardware' && filters.hardwareMode==='platform' ? config.platforms.includes(filters.hardware)||config.hardware===filters.hardware : config[field]===filters[field]));
  }
  function configurationMatches(config,filters) {
    return !(filters.version&&!filters.library) && configurationDifferences(config,filters).length===0;
  }
  function assessCoverage(index,filters) {
    // Coverage ignores display-category and keyword filters. A case association
    // is not proof that every candidate was validated in the selected runtime.
    const request=Object.fromEntries(configurationFields.map(k=>[k,filters[k]||'']));
    request.hardwareMode=filters.hardwareMode;
    const selected=configurationFields.filter(k=>request[k]);
    const exact=filterCases(index,request);
    const complete=['hardware','model','library'].every(k=>request[k]);
    const workflowOnly=exact.length>0&&exact.every(c=>c.category==='workflow-benchmark');
    const status=!selected.length?'browse':!exact.length?'unrecorded':workflowOnly?'workflow-only':complete?'recorded':'partial-selection';
    const related=exact.length||selected.length<2?[]:index.cases.filter(c=>c.category!=='workflow-benchmark').map(c=>{
      const configs=c.configurations.map(config=>({config,missing:configurationDifferences(config,request)})).sort((a,b)=>a.missing.length-b.missing.length);
      return {case:c,...configs[0],matched:selected.length-(configs[0]?.missing.length||0)};
    }).filter(row=>row.matched>0&&row.missing.length>0).sort((a,b)=>b.matched-a.matched||a.case.row.id.localeCompare(b.case.row.id)).slice(0,3);
    return {status,complete,exact,related,selected};
  }
  function recommendConfiguration(index,filters) {
    if(index.search){
      const result=retrieve(index,filters,{allow_alternatives:!!filters.allowAlternatives}).recommendation;if(!result)return null;
      const c=index.cases.find(c=>c.row.id===result.case_id),raw=result.configuration;
      const config=c.configurations.find(config=>config.recordId===raw.record.id&&config.kind===raw.record.kind&&
        `${config.kind}:${config.recordId}:${config.sideId??0}`===raw.id);
      const canonical=name=>index.search.aliases.library[choiceKey(name)]||name;
      return {case:c,config,record:index.records.get(key(raw.record.kind,raw.record.id)),
        library:config.libraries.find(lib=>canonical(lib.name)===canonical(filters.library)),
        solution:result.candidate_id?index.records.get(key('candidate',result.candidate_id)):null,
        verification:result.verification_id?index.records.get(key('verification',result.verification_id)):null,
        scope:result.scope,differences:result.differences};
    }
    if (!['hardware','model','library'].every(field=>filters[field])) return null;
    // Hardware, model, library and task are fixed. Only the requested version
    // may fall back to a different recorded version of the same library.
    const stack={...filters,version:'',category:'',q:''},options=[];
    for (const c of index.cases.filter(c=>c.category!=='workflow-benchmark')) {
      for (const config of c.configurations.filter(config=>configurationMatches(config,stack))) {
        const record=c.observations.find(o=>o.kind===config.kind&&o.record.id===config.recordId).record;
        const library=config.libraries.find(lib=>lib.name===filters.library);
        const versionKnown=lib=>typeof lib.commit==='string'&&/^[a-f0-9]{40}$/i.test(lib.commit)||
          typeof lib.version==='string'&&!!lib.version.trim()&&!/^(unknown|not[- ]recorded|unspecified|latest|main|master|head|selected source|未记录|未知)$/i.test(lib.version.trim());
        if (!config.libraries.every(versionKnown)) continue;
        const add=(solution,verification)=>options.push({case:c,config,record,library,solution,verification,
          scope:verification?.scope||record.summary,
          differences:configurationDifferences(config,filters)});
        if (config.kind==='comparison') {
          // Collection alone is not a recommendation. Only an explicit whole-
          // record "passed" decision is supported; unknown acceptance stays out.
          const rows=record.comparisons?.filter(row=>row.side_ids?.includes(config.sideId))||[];
          if (record.status==='completed'&&record.acceptance==='passed'&&record.scope_status!=='partial'&&record.whole_task_completed!==false&&
              rows.length&&rows.every(row=>row.metrics?.length&&Array.isArray(row.coverage)&&row.coverage.length&&
                row.coverage.some(count=>Number.isFinite(count.expected)&&count.expected>0)&&
                row.coverage.every(count=>Number.isFinite(count.observed)&&Number.isFinite(count.expected)&&count.expected>=0&&count.observed===count.expected))) add(null,null);
          continue;
        }
        if(record.status!=='validated'||!record.validation?.metrics?.length||!record.validation?.checks?.length||
            !Number.isFinite(record.validation.coverage?.expected)||record.validation.coverage.expected<=0||
            record.validation.coverage.observed!==record.validation.coverage.expected)continue;
        for (const entry of c.entries.filter(e=>e.kind==='candidate')) {
          const solution=index.records.get(key(entry.kind,entry.id)),d=solution.details;
          if (d.evidence_level!=='runtime-tested-in-scope'||['not-selected','selected-for-further-study'].includes(d.decision)||d.target.library!==filters.library) continue;
          if (d.target.version&&d.target.version!==library.version) continue;
          if (d.target.commit&&d.target.commit!==library.commit) continue;
          const verification=c.entries.filter(e=>e.kind==='verification').map(e=>index.records.get(key(e.kind,e.id))).find(v=>
            v.details.outcome==='scoped-checks-passed'&&
            v.details.sources.some(s=>s.kind==='repair'&&s.id===record.id&&s.pointer==='/validation/metrics')&&
            index.relations.links.some(l=>l.relation==='uses_workload'&&l.from.kind==='verification'&&l.from.id===v.id&&l.to.kind==='workload'&&
              index.records.get(key('workload',l.to.id))?.details.sources.some(s=>s.kind==='repair'&&s.id===record.id&&s.pointer==='/environment'))&&
            index.relations.links.some(l=>l.relation==='verifies'&&l.from.kind==='verification'&&l.from.id===v.id&&l.to.kind==='candidate'&&l.to.id===solution.id));
          if (verification) add(solution,verification);
        }
      }
    }
    return options.sort((a,b)=>a.differences.length-b.differences.length||Number(!!b.solution)-Number(!!a.solution)||
      (b.record.date||'').localeCompare(a.record.date||'')||a.record.id.localeCompare(b.record.id)||a.config.hardware.localeCompare(b.config.hardware))[0]||null;
  }
  function configurationIssues(index,filters) {
    if (!['hardware','model','library'].every(field=>filters[field])) return [];
    const found=new Map();
    for (const c of filterCases(index,{...filters,category:'',q:''}).filter(c=>c.category!=='workflow-benchmark')) {
      for (const entry of c.entries.filter(e=>e.kind==='issue')) {
        const issue=index.records.get(key(entry.kind,entry.id));
        const documented=index.relations.links.filter(l=>l.relation==='documents'&&l.from.kind==='issue'&&l.from.id===issue.id&&l.to.kind==='repair').map(l=>l.to.id);
        const config=c.configurations.find(config=>config.kind==='repair'&&documented.includes(config.recordId)&&configurationMatches(config,filters));
        if (!config||found.has(issue.id)) continue;
        const record=index.records.get(key('repair',config.recordId));
        const targets=index.relations.links.filter(l=>l.relation==='addresses'&&l.to.kind==='issue'&&l.to.id===issue.id&&l.from.kind==='candidate').map(l=>index.records.get(key('candidate',l.from.id))?.details.target.library).filter(Boolean);
        found.set(issue.id,{issue,case:c,config,record,libraries:[...new Set(record.affected_library?[record.affected_library]:targets)]});
      }
    }
    return [...found.values()].map(row=>({...row,resolution:issueResolution(index,row)}));
  }
  function issueResolution(index, {issue,case:c,config,record}) {
    // Resolution belongs to this issue and runtime, never to the whole case.
    if(issue.details.classification==='unconfirmed-hypothesis')return {status:'unconfirmed'};
    const links=index.relations.links;
    const linked=(relation,fromKind,fromId,toKind,toId)=>links.some(l=>l.relation===relation&&
      l.from.kind===fromKind&&l.from.id===fromId&&l.to.kind===toKind&&l.to.id===toId);
    const canonical=name=>index.search?.aliases.library[choiceKey(name)]||name;
    let partial=null;
    for(const entry of c.entries.filter(e=>e.kind==='candidate')) {
      const solution=index.records.get(key('candidate',entry.id)),d=solution.details;
      if(!linked('addresses','candidate',solution.id,'issue',issue.id)||d.evidence_level!=='runtime-tested-in-scope'||d.decision==='not-selected')continue;
      const library=config.libraries.find(lib=>canonical(lib.name)===canonical(d.target.library)&&
        (!d.target.version||d.target.version===lib.version)&&(!d.target.commit||d.target.commit===lib.commit));
      if(!library)continue;
      let support=config.searchConfig?.recommendations.find(r=>r.candidate_id===solution.id);
      if(!index.search){
        const isolated={...index,cases:[{...c,configurations:[config],entries:c.entries.filter(e=>e.kind!=='candidate'||e.id===solution.id)}]};
        const result=recommendConfiguration(isolated,{hardware:config.hardware,model:config.model,library:library.name,version:library.version,task:config.tasks[0]});
        if(result?.solution?.id===solution.id)support={verification_id:result.verification.id};
      }
      if(support)return {status:'fixed',solution,verification:index.records.get(key('verification',support.verification_id))};
      const verification=c.entries.filter(e=>e.kind==='verification').map(e=>index.records.get(key('verification',e.id))).find(v=>
        v.details.outcome==='partial'&&linked('verifies','verification',v.id,'candidate',solution.id)&&
        v.details.sources.some(s=>s.kind==='repair'&&s.id===record.id&&['/validation/metrics','/validation/acceptance'].includes(s.pointer))&&
        links.some(l=>l.relation==='uses_workload'&&l.from.kind==='verification'&&l.from.id===v.id&&l.to.kind==='workload'&&
          index.records.get(key('workload',l.to.id))?.details.sources.some(s=>
            s.kind==='repair'&&s.id===record.id&&s.pointer==='/environment'||
            s.file&&s.sha256&&v.details.sources.some(vs=>vs.file===s.file&&vs.sha256===s.sha256))));
      if(verification)partial={status:'partial',solution,verification};
    }
    // Partial checks establish neither complete resolution nor universal improvement.
    return partial||{status:'unresolved'};
  }
  function issueSolutions(index,c,filters,issueId) {
    const issue=c.entries.some(e=>e.kind==='issue'&&e.id===issueId)?index.records.get(key('issue',issueId)):null;
    if(!issue)return null;
    const row=configurationIssues(index,filters).find(r=>r.issue.id===issueId&&r.case.row.id===c.row.id);
    const resolution=row?.resolution||{status:issue.details.classification==='unconfirmed-hypothesis'?'unconfirmed':'unresolved'};
    const solutions=c.entries.filter(e=>e.kind==='candidate'&&index.relations.links.some(l=>l.relation==='addresses'&&
      l.from.kind==='candidate'&&l.from.id===e.id&&l.to.kind==='issue'&&l.to.id===issueId)).map(e=>index.records.get(key('candidate',e.id)));
    const primary=resolution.status==='fixed'?resolution.solution:null;
    return {issue,resolution,primary,config:row?.config,alternatives:solutions.filter(s=>s.id!==primary?.id)};
  }
  let loading;
  let solutionResultIndex;
  const solutionEvidence=new Map();
  async function fetchCaseData(path) {
    const response=await fetch('./'+path,{cache:'no-cache'});
    if(!response.ok)throw new Error('HTTP '+response.status);
    return response;
  }
  function loadSolutionResults() {
    return solutionResultIndex??=(fetchCaseData('data/solution-results.json').then(response=>response.json())
      .catch(error=>{solutionResultIndex=null;throw error;}));
  }
  function loadSolutionEvidence(file,sha256) {
    if(!/^evidence\/[a-z0-9][a-z0-9._-]*\.json$/i.test(file)||!sha256)return Promise.reject(new Error('Missing evidence pin'));
    const cacheKey=key(file,sha256);
    if(!solutionEvidence.has(cacheKey))solutionEvidence.set(cacheKey,fetchCaseData(file).then(async response=>{
      const bytes=await response.arrayBuffer();
      const hash=Array.from(new Uint8Array(await global.crypto.subtle.digest('SHA-256',bytes))).map(v=>v.toString(16).padStart(2,'0')).join('');
      if(hash!==sha256)throw new Error('Evidence hash mismatch');
      return JSON.parse(new TextDecoder().decode(bytes));
    }).catch(error=>{solutionEvidence.delete(cacheKey);throw error;}));
    return solutionEvidence.get(cacheKey);
  }
  function pointerValue(doc,pointer) {
    return pointer.split('/').slice(1).reduce((value,part)=>{
      const field=part.replace(/~1/g,'/').replace(/~0/g,'~');
      return value!=null&&Object.hasOwn(value,field)?value[field]:undefined;
    },doc);
  }
  const sourceCovers=(source,pointer)=>source.pointer===pointer||pointer.startsWith(source.pointer+'/');
  const reportLink=(c,kind,id,filters={},issue='',solution='')=>{
    const params=new URLSearchParams({[kind]:id});
    for(const field of configurationFields)if(filters[field])params.set(field,filters[field]);
    if(issue)params.set('issue',issue);
    if(solution)params.set('solution',solution);
    return `#/report/${encodeURIComponent(c.row.id)}?${params}`;
  };
  const verificationBindings=(index,c)=>{
    const members=new Set(c.entries.map(e=>key(e.kind,e.id)));
    return index.relations.links.flatMap(link=>link.relation==='verifies'&&link.from.kind==='verification'&&link.to.kind==='candidate'&&
      members.has(key('verification',link.from.id))&&members.has(key('candidate',link.to.id))?
      [{verification:index.records.get(key('verification',link.from.id)),solution:index.records.get(key('candidate',link.to.id))}]:[]);
  };
  const caseRepairs=c=>new Map(c.observations.filter(o=>o.kind==='repair').map(o=>[o.record.id,o.record]));
  async function resolveResultGroup(group,bindings,repairs,tx) {
    const rows=bindings.filter(({verification})=>Object.hasOwn(group.results,verification.id));
    if(!rows.length)return null;
    const pins=new Set(rows.flatMap(({verification})=>verification.details.sources.filter(s=>s.file===group.file).map(s=>s.sha256)));
    if(pins.size!==1)throw new Error('Expected one evidence pin');
    const evidence=await loadSolutionEvidence(group.file,[...pins][0]);
    const columns=rows.map(({solution,verification})=>{
      const spec=group.results[verification.id],sources=verification.details.sources;
      const metrics=group.metrics.map(({label,unit,path,recordMetric,relative})=>{
        const before=relative?1:pointerValue(evidence,group.baseline+path);
        let after;
        if(spec.repair&&sources.some(s=>s.kind==='repair'&&s.id===spec.repair&&s.pointer==='/validation/metrics')){
          const metric=repairs.get(spec.repair)?.validation?.metrics?.find(m=>m.id===recordMetric);
          if(metric?.before===before)after=metric.after;
        }else if(spec.pointer&&sources.some(s=>s.file===group.file&&sourceCovers(s,spec.pointer+path))){
          after=pointerValue(evidence,spec.pointer+path);
        }
        return {label:tx(label.zh,label.en),unit,before,after};
      });
      if(!metrics.every(m=>Number.isFinite(m.before)&&Number.isFinite(m.after)))throw new Error('Incomplete candidate measurements');
      return {solution,verification,metrics};
    });
    return {label:tx(group.label.zh,group.label.en),columns};
  }
  async function loadIndex() {
    if (!loading) loading = Promise.all(['relations','evaluations','repairs','content','frameworks','search-index'].map(async name => {
      const response = await fetch('./data/'+name+'.json',{cache:'no-cache'});
      if (!response.ok) throw new Error(name+': HTTP '+response.status);
      const text=await response.text();return {name,text,document:JSON.parse(text)};
    })).then(async entries=>{
      const metadata=entries.at(-1).document;
      for(const entry of entries.slice(0,-1)){
        const pin=metadata.sources.find(source=>source.file==='data/'+entry.name+'.json');
        const hash=Array.from(new Uint8Array(await global.crypto.subtle.digest('SHA-256',new TextEncoder().encode(entry.text)))).map(v=>v.toString(16).padStart(2,'0')).join('');
        if(!pin||pin.sha256!==hash)throw new Error('Search metadata is stale: '+entry.name);
      }
      return buildIndex(...entries.slice(0,3).map(entry=>entry.document),metadata);
    }).catch(error => {loading=null; throw error;});
    return loading;
  }
  async function mount(root, hash, options={}) {
    const tx=(zh,en)=>options.lang==='en'?en:zh;
    root._lifecycleCleanup?.();
    const stamp={};root._lifecycleRequest=stamp;
    root._lifecycleCleanup=()=>{root._lifecycleRequest=null;};
    root.innerHTML=`<div class="wrap lc-loading" role="status">${tx('正在读取案例…','Loading cases…')}</div>`;
    let index;
    try {index=await loadIndex();} catch(error) {
      if(root._lifecycleRequest!==stamp)return;
      root.innerHTML=`<div class="wrap lc-error" role="alert"><h1>${tx('案例库暂时无法读取','Case library unavailable')}</h1><p>${esc(error.message)}</p><button class="btn" data-retry>${tx('重试','Retry')}</button></div>`;
      root.querySelector('[data-retry]').onclick=()=>mount(root,hash,options);return;
    }
    if(root._lifecycleRequest!==stamp)return;
    const state=Object.fromEntries(configurationFields.map(k=>[k,hash.q.get(k)||'']));
    try{Object.assign(state,Search.normalizeRequest(Object.fromEntries(configurationFields.map(k=>[k,state[k]])),index.search));delete state.query;}
    catch(error){root.innerHTML=`<div class="wrap lc-error"><h1>${tx('请核对检索条件','Check the search request')}</h1><p>${esc(error.message)}</p><a href="#/cases">${tx('清除条件重新选择','Clear filters and choose again')}</a></div>`;return;}
    state.q='';
    let allowAlternatives=hash.q.get('alternatives')==='1';
    state.hardwareMode='platform';
    let selected=hash.arg||hash.q.get('id')||'',selectedIssue=hash.q.get('issue')||'';
    if(!state.library)state.version='';
    const get=(kind,id)=>index.records.get(key(kind,id));
    const description=(kind,row)=>Search.describe(index,kind,row.id,options.lang||'zh',
      {title:row.details?.title||row.title||row.id});
    const badge=(text,cls='')=>`<span class="lc-badge ${cls}">${esc(text)}</span>`;
    const verificationLabels={
      'scoped-checks-passed':tx('检查通过','Checks passed'),partial:tx('部分检查通过','Some checks passed'),
      'mixed-results':tx('指标有取舍','Mixed results'),'not-adopted':tx('未采用','Not adopted'),
      'not-runtime-tested':tx('未运行验证','Not runtime tested'),
      'further-testing':tx('待进一步验证','Further testing planned'),
      'no-verification':tx('暂无验证记录','No verification recorded')};
    const verificationBadge=outcome=>{
      const passed=outcome==='scoped-checks-passed',untested=outcome==='not-runtime-tested';
      const icon=passed?'<path d="m3 7 3 3 5-6"/>':untested?'<circle cx="7" cy="7" r="5"/><path d="M7 4v3l2 1"/>':'';
      const tone=passed?' lc-verification-passed':untested?' lc-verification-untested':'';
      return `<span class="lc-verification${tone}">${icon?`<svg viewBox="0 0 14 14" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${icon}</svg>`:''}${esc(verificationLabels[outcome]||outcome)}</span>`;
    };
    const resolutionBadge=r=>badge({fixed:tx('有实测方案','Tested fix'),partial:tx('部分验证','Partially verified'),unresolved:tx('待修复','Open'),unconfirmed:tx('待确认','Unconfirmed')}[r.status],`lc-resolution lc-resolution-${r.status}`);
    const resolutionNote=r=>r.status==='fixed'?(['patch','backport'].includes(r.solution.details.solution_type)?tx('需本地补丁','Local patch required'):''):
      '';
    const unique=values=>[...new Set(values.filter(v=>v!==null&&v!==undefined&&v!==''))].sort((a,b)=>String(a).localeCompare(String(b)));
    const catalogCases=index.cases.filter(c=>c.category!=='workflow-benchmark');
    const configs=catalogCases.flatMap(c=>c.configurations);
    const canonical=(field,value)=>index.search.aliases[field]?.[choiceKey(value)]||value;
    const example=index.search.cases.flatMap(c=>c.configurations.filter(config=>config.recommendations.length))
      .sort((a,b)=>(b.date||'').localeCompare(a.date||'')||a.id.localeCompare(b.id))
      .flatMap(config=>config.libraries.filter(lib=>config.frameworks.includes(canonical('library',lib.name))&&
        config.recommendations.some(support=>!support.library_keys||support.library_keys.some(k=>lib.keys.includes(k))))
        .map(lib=>({hardware:config.platforms[0],model:config.model,model_size:'',library:canonical('library',lib.name),version:lib.version||'',task:config.tasks[0]||''})))
      .find(request=>recommendConfiguration(index,request));
    let showingExample=!!example&&hash.q.get('example')==='1'&&configurationFields.every(field=>state[field]===example[field]);
    if(example&&!selected&&!hash.q.size){Object.assign(state,example);showingExample=true;}
    const labels={hardware:tx('硬件平台','Hardware'),model:tx('模型','Model'),model_size:tx('参数量','Parameter count'),library:tx('框架','Framework'),task:tx('任务','Task'),version:tx('框架版本','Framework version')};
    const selectorKeys=['hardware','model','library','version'];
    const frameworkGroups=[['sft',tx('训练','Training')],['inference',tx('推理','Inference')],['rl','RL']];
    const taskLabel=value=>({inference:tx('推理','Inference'),sft:'SFT',rl:'RL'}[value]||value);
    const display=(field,value)=>field==='model'&&value==='unclassified'?tx('未分类','Unclassified'):
      field==='model_size'?({tiny:tx('Tiny / 自定义','Tiny / custom'),unknown:tx('未记录','Not recorded')}[value]||
        (/^\d+$/.test(value)?new Intl.NumberFormat().format(BigInt(value)):value)):value;
    function libraryVersion(lib) {
      const {number,kind}=versionParts(lib);
      if(kind==='source')return number?`${number} · ${tx('源码','Source')}`:tx('源码版本','Source build');
      return kind==='build'?`${number} · ${tx('构建版','Build')}`:number;
    }
    const requestedVersion=value=>libraryVersion({version:value})||tx('指定版本','Selected version');
    root.innerHTML=`<div class="wrap lc-page">
      <div class="lc-intro"><h1>${tx('精度案例库','Precision Case Library')}</h1></div>
      <div class="lc-discovery"><aside class="lc-demand" aria-label="${tx('目标配置','Target configuration')}"><div class="lc-demand-head"><h2>${tx('目标配置','Target configuration')}</h2><button type="button" class="link" data-reset>${tx('重置','Reset')}</button></div><div class="lc-selectors"></div></aside>
      <section class="lc-results" aria-label="${tx('配置结果','Configuration results')}"><section class="lc-recommendation" aria-live="polite"></section><section class="lc-problems"><h2>${tx('相关问题','Related issues')}</h2><div class="lc-list"></div></section><section class="lc-problems" data-model-records hidden></section></section></div>
      <dialog class="lc-option-dialog" aria-labelledby="lc-option-title"><div class="lc-modal-bar"><h2 id="lc-option-title"></h2><button type="button" class="iconbtn" data-options-close aria-label="${tx('关闭选项','Close options')}">×</button></div><div class="lc-option-body"><label class="lc-option-search"><svg class="lc-search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.5 4.5"/></svg><span class="sr">${tx('搜索选项','Search options')}</span><input type="search" data-options-search placeholder="${tx('搜索或输入名称','Search or enter a name')}"></label><div class="lc-option-grid" data-options-list></div></div></dialog>
      <dialog class="lc-case-dialog" aria-labelledby="lc-case-title"><div class="lc-modal-bar"><span>${tx('案例详情','Case detail')}</span><div class="lc-detail-actions"><button class="btn sm" type="button" data-copy>${tx('复制链接','Copy link')}</button><button type="button" class="iconbtn" data-detail-close aria-label="${tx('关闭案例详情','Close case detail')}">×</button></div></div><article class="lc-detail" id="lc-detail"></article></dialog></div>`;
    const list=root.querySelector('.lc-list'),detail=root.querySelector('.lc-detail'),dialog=root.querySelector('.lc-case-dialog'),optionDialog=root.querySelector('.lc-option-dialog');
    const previewWidths=new WeakMap();
    const previewResize=new ResizeObserver(entries=>{
      for(const {target} of entries){
        const width=target.clientWidth;
        if(width===previewWidths.get(target))continue;
        previewWidths.set(target,width);
        fitPreviewChoices(target);
      }
    });
    let cleanup=false,lastTrigger=null,activeField='',optionTrigger=null;
    const updateLock=()=>document.body.classList.toggle('locked',dialog.open||optionDialog.open);
    root._lifecycleCleanup=()=>{cleanup=true;previewResize.disconnect();root._lifecycleRequest=null;if(optionDialog.open)optionDialog.close();if(dialog.open)dialog.close();updateLock();};
    document.fonts.ready.then(()=>{if(!cleanup)root.querySelectorAll('[data-preview-rows]').forEach(fitPreviewChoices);});
    const closeDetail=()=>{if(dialog.open)dialog.close();};
    dialog.addEventListener('close',()=>{if(cleanup)return;updateLock();selected='';selectedIssue='';writeURL();list.querySelectorAll('[data-case]').forEach(b=>b.setAttribute('aria-expanded','false'));if(lastTrigger?.isConnected)lastTrigger.focus({preventScroll:true});});
    root.querySelector('[data-detail-close]').onclick=closeDetail;
    const copyLink=dialog.querySelector('[data-copy]');
    copyLink.onclick=async()=>{try{await navigator.clipboard.writeText(location.href);copyLink.textContent=tx('已复制','Copied');}catch(_){copyLink.textContent=tx('请复制地址栏链接','Copy the address-bar URL');}};
    root.querySelector('[data-options-close]').onclick=()=>optionDialog.close();
    optionDialog.addEventListener('close',()=>{
      if(cleanup||optionDialog.open)return;
      updateLock();
      if(optionTrigger?.isConnected)optionTrigger.focus({preventScroll:true});else focusChoice(activeField);
      optionTrigger=null;
    });
    for(const modal of [dialog,optionDialog])modal.addEventListener('click',event=>{const rect=modal.getBoundingClientRect();if(event.target===modal&&(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom))modal.close();});
    function valuesFor(field) {
      if(field==='hardware')return unique(configs.flatMap(c=>c.platforms));
      if(field==='model')return unique(configs.map(c=>c.searchConfig?.model_group??c.searchConfig?.model_architecture));
      if(field==='version')return unique(configs.filter(c=>c.searchConfig?.frameworks.includes(state.library)&&
        (!state.task||c.tasks.includes(state.task)))
        .flatMap(c=>c.libraries.filter(lib=>canonical('library',lib.name)===state.library).map(lib=>lib.version)));
      return [];
    }
    const rankedChoices=(field,values,task='')=>values.map(value=>({value,count:filterCases({cases:catalogCases},
      {hardwareMode:state.hardwareMode,task,[field]:value}).length}))
      .sort((a,b)=>b.count-a.count||a.value.localeCompare(b.value)).map(row=>row.value);
    const previews=Object.fromEntries(['hardware','model'].map(field=>[field,rankedChoices(field,valuesFor(field)).slice(0,field==='hardware'?5:4)]));
    const frameworkCatalog=new Map(frameworkGroups.map(([task])=>[task,unique(configs.filter(c=>c.tasks.includes(task)).flatMap(c=>c.searchConfig?.frameworks||[]))]));
    const frameworkPreviews=new Map([...frameworkCatalog].map(([task,values])=>[task,rankedChoices('library',values,task)]));
    const selectionLabel=text=>text?`<span class="lc-selection" title="${esc(text)}">${esc(text)}</span>`:'';
    function hardwareMark(value) {
      const vendor=choiceGroup('hardware',value)?.[0];
      if(vendor==='nvidia')return '<img class="lc-vendor-logo" src="./assets/nvidia.svg" alt="" aria-hidden="true" width="22" height="22">';
      if(vendor==='ascend')return '<img class="lc-vendor-logo lc-ascend-logo" src="./assets/ascend.svg" alt="" aria-hidden="true" width="30" height="22">';
      if(value==='CPU')return '<svg class="lc-vendor-logo" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M8 1v4m8-4v4M8 19v4m8-4v4M1 8h4m-4 8h4m14-8h4m-4 8h4"/></svg>';
      return '';
    }
    function modelMark(value) {
      if(!value)return '';
      const family=choiceGroup('model',value)?.[0];
      if(family==='nvidia')return hardwareMark('NVIDIA');
      if(family)return `<svg class="lc-model-logo" aria-hidden="true" viewBox="0 0 24 24" fill="currentColor"><use href="./assets/model-logos.svg#${family}"></use></svg>`;
      return '<svg class="lc-model-logo" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9ZM4 7.5l8 4.5 8-4.5M12 12v9"/></svg>';
    }
    function moreChoice(field) {
      return `<button class="pill lc-choice lc-more" type="button" data-selector-more="${field}" aria-label="${tx('更多','More')} ${labels[field]}" aria-haspopup="dialog" title="${tx('更多选项','More options')}">…</button>`;
    }
    function bindChoices(box,field) {
      box.querySelectorAll('[data-choose-field]').forEach(button=>button.onclick=()=>choose(field,button.dataset.value,button.dataset.task));
      box.querySelector('[data-selector-more]')?.addEventListener('click',event=>openOptions(field,event.currentTarget));
    }
    function fitPreviewChoices(row) {
      const maxRows=Number(row.dataset.previewRows);
      const buttons=[...row.children],toggle=buttons.pop();
      for(const button of buttons)button.hidden=false;
      toggle.hidden=true;
      const tops=buttons.map(button=>button.offsetTop);
      const rows=[...new Set(tops)];
      if(rows.length>maxRows){
        const lastRow=rows[maxRows-1];
        buttons.forEach((button,i)=>{button.hidden=tops[i]>lastRow;});
        toggle.hidden=false;
        // Make room for the overflow button on the last visible row.
        let last=tops.lastIndexOf(lastRow);
        while(last>=0&&toggle.offsetTop>lastRow)buttons[last--].hidden=true;
      }
      const summary=row.previousElementSibling?.querySelector('.lc-selection');
      if(summary)summary.hidden=buttons.some(button=>!button.hidden&&button.getAttribute('aria-pressed')==='true');
    }
    function paintFrameworks(box,query,expanded) {
      const term=choiceKey(query),alias=choiceKey(canonical('library',query));
      const choice=(task,library,label=library)=>`<button class="pill lc-choice" type="button" data-choose-field="library" data-task="${task}" data-value="${esc(library)}" aria-pressed="${state.task===task&&state.library===library}"><span>${esc(label)}</span></button>`;
      const matches=list=>list.filter(name=>!term||choiceKey(name)===alias||choiceKey(name).includes(term));
      const groups=frameworkGroups.map(([task,label])=>{
        const names=matches((expanded?frameworkCatalog:frameworkPreviews).get(task));
        const all=expanded&&!term?choice(task,'',tx('全部','All')):'';
        if(!names.length&&!all)return '';
        const summary=!expanded&&state.task===task?selectionLabel(state.library||tx('全部','All')):'';
        const more=tx(`更多${label}框架`,`More ${label} frameworks`);
        const toggle=expanded?'':`<button class="pill lc-choice lc-more" type="button" data-framework-toggle="${task}" aria-haspopup="dialog" aria-label="${more}" title="${more}" hidden>…</button>`;
        return `<section class="lc-framework-group" data-framework-task="${task}" aria-label="${label}"><div class="lc-field-label"><h3>${label}</h3>${summary}</div><div class="pills"${expanded?'':' data-preview-rows="3"'}>${all}${names.map(name=>choice(task,name)).join('')}${toggle}</div></section>`;
      }).join('');
      const top=expanded&&(state.library||state.task)?choice('','',tx('清除框架筛选','Clear framework filter')):'';
      box.innerHTML=(top?`<div class="pills">${top}</div>`:'')+groups+
        (query&&!groups?`<p class="lc-placeholder">${tx('没有匹配的框架','No matching frameworks')}</p>`:'');
      if(expanded)box.scrollTop=0;
      bindChoices(box,'library');
      for(const row of box.querySelectorAll('[data-preview-rows]')){
        const toggle=row.lastElementChild;
        toggle.onclick=()=>openOptions('library',toggle);
        fitPreviewChoices(row);
        previewResize.observe(row);
      }
    }
    function paintChoices(field,expanded=false) {
      const box=expanded?root.querySelector('[data-options-list]'):root.querySelector(`[data-choices="${field}"]`);
      const query=expanded?root.querySelector('[data-options-search]').value.trim():'';
      if(field==='library')return paintFrameworks(box,query,expanded);
      if(field==='version'){
        const labels=new Map();
        for(const value of valuesFor(field)){
          if(!versionParts({version:value}).number)continue;
          const label=libraryVersion({version:value});
          labels.set(label,[...(labels.get(label)||[]),value]);
        }
        const term=choiceKey(query);
        const choice=(value,label)=>`<button class="pill lc-choice" type="button" data-choose-field="version" data-value="${esc(value)}" aria-pressed="${state.version===value}">${esc(label)}</button>`;
        const versions=[...labels].filter(([label,values])=>values.length===1&&(!term||choiceKey(label).includes(term)));
        const clear=choice('',tx('不限','Any'));
        const choices=versions.map(([label,[value]])=>choice(value,label)).join('');
        if(!expanded){
          if(!state.library||!versions.length&&!state.version){
            box.innerHTML=`<span class="lc-version-empty">${state.library?tx('暂无可选版本号','No version numbers available'):tx('先选择框架','Select a framework first')}</span>`;
            return;
          }
          box.innerHTML=clear+choices+(versions.length?moreChoice(field):'');
          bindChoices(box,field);
          if(versions.length){
            box.dataset.previewRows='1';
            fitPreviewChoices(box);
            previewResize.observe(box);
          }
          return;
        }
        const custom=releaseVersion.test(query)&&![...labels.values()].flat().some(value=>
          choiceKey(versionParts({version:value}).number)===choiceKey(query.replace(/^v/i,'')))?choice(query,tx('使用','Use')+' '+requestedVersion(query)):'';
        box.innerHTML=`<div class="pills">${clear}${choices}${custom}</div>`+
          (!versions.length&&!custom?`<p class="lc-placeholder">${query?tx('没有匹配的版本','No matching versions'):tx('暂无可选版本号','No version numbers available')}</p>`:'');
        box.scrollTop=0;
        bindChoices(box,field);
        return;
      }
      const all=expanded?valuesFor(field):previews[field];
      const term=choiceKey(query);
      const platforms=field==='hardware'&&query?Search.hardwarePlatforms(query,index.search):[];
      const alias=canonical(field,platforms.length===1?platforms[0]:query);
      const sameChoice=(a,b)=>choiceKey(a)===choiceKey(b);
      const values=all.filter(value=>!term||sameChoice(value,alias)||choiceKey(value).includes(term)||
        choiceKey(display(field,value)).includes(term)||
        choiceKey(choiceGroup(field,value)?.[1]||'').includes(term));
      if(expanded&&state[field]&&!values.includes(state[field])&&!query&&field!=='model')values.push(state[field]);
      const choice=(value,label=display(field,value))=>`<button class="pill lc-choice" type="button" data-choose-field="${field}" data-value="${esc(value)}" aria-pressed="${state[field]===value}">${field==='hardware'?hardwareMark(value):field==='model'?modelMark(value):''}<span>${esc(label)}</span></button>`;
      const clear=choice('',tx('不限','Any'));
      const exactModels=field==='model'&&expanded?unique(configs.map(c=>c.searchConfig).filter(c=>{
        if(!c?.model||all.some(value=>sameChoice(value,c.model)))return false;
        return term?[c.model,...(c.keys.model_identity??[])].some(name=>choiceKey(name).includes(term)):
          c.model_size!=='tiny'&&state.model&&c.keys.model.includes(choiceKey(state.model));
      }).map(c=>c.model)):[];
      const custom=query&&![...all,...exactModels].some(value=>sameChoice(value,alias))?choice(query,tx('使用','Use')+' '+query):'';
      if(expanded&&choiceGroups[field]){
        const groups=new Map(choiceGroups[field].map(([id,label])=>[id,{label,values:[]}]));
        groups.set('other',{label:field==='hardware'?tx('其他硬件','Other hardware'):tx('其他模型','Other models'),values:[]});
        for(const value of values)groups.get(choiceGroup(field,value)?.[0]||'other').values.push(value);
        box.innerHTML=`<div class="pills">${clear}${custom}</div>`+[...groups].filter(([, group])=>group.values.length)
          .map(([id, group])=>`<section class="lc-option-group" aria-labelledby="lc-group-${id}"><h3 id="lc-group-${id}">${esc(group.label)}</h3><div class="pills">${group.values.map(value=>choice(value)).join('')}</div></section>`).join('');
        if(exactModels.length)box.insertAdjacentHTML('beforeend',`<section class="lc-option-group" aria-labelledby="lc-exact-models"><h3 id="lc-exact-models">${tx('具体型号','Specific models')}</h3><div class="pills">${exactModels.map(value=>choice(value)).join('')}</div></section>`);
      }else{
        const buttons=clear+values.map(value=>choice(value)).join('')+custom;
        box.innerHTML=expanded?`<div class="pills">${buttons}</div>`:buttons+moreChoice(field);
      }
      if(expanded)box.scrollTop=0;
      bindChoices(box,field);
    }
    function refreshSelectors() {
      previewResize.disconnect();
      root.querySelector('.lc-selectors').innerHTML=selectorKeys.map((field,i)=>{
        const hasValue=state[field]||field==='library'&&state.task;
        const outsidePreview=previews[field]&&hasValue&&!previews[field].includes(state[field]);
        const summary=field==='version'&&state.version?requestedVersion(state.version):
          field==='model'&&state.model_size?[state.model,display('model_size',state.model_size)].filter(Boolean).join(' · '):
          outsidePreview||field==='library'&&!state.task?state[field]:'';
        const controls=`<div class="${field==='library'?'lc-framework-options':'pills'}" data-choices="${field}"></div>`;
        return `<fieldset class="lc-select-field${hasValue?' has-value':''}"><legend><span class="lc-field-label"><span><span class="lc-field-step">0${i+1}</span>${labels[field]}${field==='version'?` <small>${tx('可选','Optional')}</small>`:''}</span>${selectionLabel(summary)}</span></legend>${controls}</fieldset>`;
      }).join('');
      for(const field of selectorKeys)paintChoices(field);
    }
    function openOptions(field,trigger=null) {
      optionTrigger=trigger;
      activeField=field;optionDialog.dataset.field=field;root.querySelector('#lc-option-title').textContent=labels[field];
      root.querySelector('[data-options-search]').value='';paintChoices(field,true);
      optionDialog.showModal();updateLock();root.querySelector('[data-options-search]').focus({preventScroll:true});
      if(trigger?.dataset.frameworkToggle){
        const list=optionDialog.querySelector('[data-options-list]');
        const group=list.querySelector(`[data-framework-task="${trigger.dataset.frameworkToggle}"]`);
        if(group)list.scrollTop=group.getBoundingClientRect().top-list.getBoundingClientRect().top+list.scrollTop;
      }
    }
    root.querySelector('[data-options-search]').oninput=()=>paintChoices(activeField,true);
    function focusChoice(field) {
      const choices=[...root.querySelectorAll(`.lc-selectors [data-choose-field="${field}"]`)].filter(button=>
        !button.hidden&&(field!=='library'||!state.task||button.dataset.task===state.task));
      const button=choices.find(button=>button.getAttribute('aria-pressed')==='true')||
        root.querySelector(`.lc-selectors [data-selector-more="${field}"]:not([hidden])`)||
        (field==='library'&&(root.querySelector(`.lc-selectors [data-framework-toggle="${state.task}"]:not([hidden])`)||
          root.querySelector('.lc-selectors [data-framework-toggle]:not([hidden])')))||choices[0];
      button?.focus({preventScroll:true});
    }
    function choose(field,value,task='') {
      const request=Object.fromEntries(configurationFields.map(k=>[k,state[k]]));request[field]=value;
      if(field==='library'){request.task=task;request.version='';}
      if(field==='model')request.model_size='';
      try{const normalized=Search.normalizeRequest(request,index.search);for(const k of configurationFields)state[k]=normalized[k];}
      catch(error){root.querySelector('[data-options-list]').innerHTML=`<p class="lc-error">${esc(error.message)}</p>`;return;}
      allowAlternatives=false;showingExample=false;
      if(optionDialog.open)optionDialog.close();selected='';render();focusChoice(field);
    }
    function writeURL() {
      const query=new URLSearchParams();
      for(const [k,v] of Object.entries(state))if(v||k==='category')query.set(k,v);
      if(allowAlternatives)query.set('alternatives','1');
      if(showingExample)query.set('example','1');
      if(selected&&selectedIssue)query.set('issue',selectedIssue);
      history.replaceState(null,'','#/cases'+(selected?'/'+encodeURIComponent(selected):'')+'?'+query.toString());
    }
    function solutionAction({solution_type:type,target}) {
      const version=libraryVersion(target),library=[target.library,version].filter(Boolean).join(' ');
      if(type==='patch'||type==='backport')return tx(`给 ${library} 打补丁`,`Apply a patch to ${library}`);
      if(type==='upgrade')return versionParts(target).number?tx(`升级到 ${library}`,`Upgrade to ${library}`):
        tx(`改用含修复的 ${target.library} 源码`,`Use the ${target.library} source with the fix`);
      if(type==='configuration')return tx(`调整 ${library} 的运行设置`,`Change ${library} runtime settings`);
      return tx(`调整 ${library} 的计算精度`,`Change how ${library} computes in different precisions`);
    }
    const metricNumber=new Intl.NumberFormat(options.lang==='en'?'en':'zh-CN',{maximumSignificantDigits:5});
    const formatMetric=n=>n!==0&&(Math.abs(n)<.001||Math.abs(n)>=1e6)?n.toExponential(3):metricNumber.format(n);
    function renderMetric({label,unit,before,after,peers=[]}) {
      const values=[before,after,...peers];
      const magnitude=Math.max(...values.map(Math.abs))||1;
      const scaled=values.map(n=>n/magnitude);
      const low=Math.min(0,...scaled),high=Math.max(0,...scaled);
      const position=n=>100*(n/magnitude-low)/(high-low||1);
      const origin=position(0);
      const point=(n,phase,text)=>{
        const end=position(n),span=Math.abs(end-origin);
        return `<div class="lc-metric-point lc-metric-${phase}"><span>${text}</span><strong>${formatMetric(n)}</strong><span class="lc-metric-track" aria-hidden="true" style="--zero:${origin}%;--start:${Math.min(origin,end)}%;--span:${span}%"><i></i></span></div>`;
      };
      return `<figure class="lc-metric"><figcaption>${esc(label)}${unit?` <small>${esc(unit)}</small>`:''}</figcaption>${point(before,'before',tx('基线','Baseline'))}${point(after,'after',tx('本方案','This option'))}</figure>`;
    }
    function showCase(c,issueId='') {
      if(!c) {detail.innerHTML=`<div class="lc-empty"><h2>${tx('没有匹配案例','No matching cases')}</h2><p>${tx('调整筛选条件，或切换分类查看其他记录。','Adjust the filters or select another category.')}</p></div>`;return;}
      const {row,entries}=c,d=row.details;
      const recommendation=recommendConfiguration(index,{...state,allowAlternatives});
      const filters=recommendation?{...state,version:recommendation.library.version||''}:state;
      const focused=issueSolutions(index,c,filters,issueId);
      const issues=focused?[]:entries.filter(e=>e.kind==='issue').map(e=>get(e.kind,e.id));
      const resolutionRows=issues.length?configurationIssues(index,filters):[];
      const solutions=focused?focused.alternatives:entries.filter(e=>e.kind==='candidate').map(e=>get(e.kind,e.id));
      const primary=focused?focused.primary||focused.resolution.solution:recommendation?.case.row.id===row.id?recommendation.solution:null;
      const solutionRows=primary?[primary,...solutions.filter(s=>s.id!==primary.id)]:solutions;
      const repairs=caseRepairs(c),allBindings=verificationBindings(index,c);
      const bindings=allBindings.filter(({solution})=>solutionRows.some(row=>row.id===solution.id));
      const linkedVerifications=solution=>bindings.filter(binding=>binding.solution.id===solution.id).map(binding=>binding.verification);
      const effects=bindings.flatMap(({solution,verification})=>{
        // Acceptance-only sources can be shared across candidates; they do not bind the metric pairs to a candidate.
        const ids=new Set(verification.details.sources.filter(s=>s.kind==='repair'&&s.pointer==='/validation/metrics').map(s=>s.id));
        return [...ids].flatMap(id=>{
          const record=repairs.get(id);
          const metrics=record?.validation?.metrics?.filter(m=>Number.isFinite(m.before)&&Number.isFinite(m.after));
          return metrics?.length?[{solution,verification,record,metrics}]:[];
        });
      }).sort((a,b)=>Number(b.record.id===focused?.config?.recordId)-Number(a.record.id===focused?.config?.recordId));
      const environmentLabel=config=>[config.platforms.join(' / ')||config.hardware,config.model,...config.tasks.map(taskLabel)].filter(Boolean).join(' · ');
      let selectedSolution=solutionRows.find(s=>s.id===hash.q.get('solution'))||solutionRows[0],groups=[],pending=bindings.length>0;
      let failedVerifications=new Set();
      const solutionDetails=new Map(solutionRows.map(solution=>{
        const sd=solution.details,copy=description('candidate',solution),action=solutionAction(sd);
        const checks=linkedVerifications(solution),outcomes=new Set(checks.map(v=>v.details.outcome));
        if(sd.decision==='not-selected')outcomes.add('not-adopted');
        if(sd.decision==='selected-for-further-study')outcomes.add('further-testing');
        if(!checks.length)outcomes.add(sd.evidence_level==='source-inspected'?'not-runtime-tested':'no-verification');
        return [solution.id,{title:copy.title||action,summary:copy.summary||(copy.title!==action?action:''),checks,outcomes}];
      }));
      const renderSolution=solution=>{
        const {title,checks,outcomes}=solutionDetails.get(solution.id);
        const matched=groups.flatMap(({label,columns})=>columns.filter(col=>col.solution.id===solution.id).map(col=>({label,
          verification:col.verification.id,metrics:col.metrics.map((metric,i)=>({...metric,peers:columns.map(peer=>peer.metrics[i].after)}))})));
        const replaced=new Set(matched.map(result=>result.verification));
        const measurements=[...matched,...effects.filter(e=>e.solution.id===solution.id&&!replaced.has(e.verification.id)).map(({record,metrics})=>({metrics,label:environmentLabel(c.configurations.find(config=>config.kind==='repair'&&config.recordId===record.id))}))];
        const results=measurements.map(({label,metrics})=>`<div class="lc-solution-measurements"><p>${esc(label)}</p><div class="lc-metrics">${metrics.map(renderMetric).join('')}</div></div>`).join('');
        const status=checks.some(v=>failedVerifications.has(v.id))?tx('部分实测结果暂不可用','Some measured results are unavailable'):
          !results?(checks.length&&pending?tx('正在读取结果…','Loading results…'):
            outcomes.has('not-runtime-tested')?verificationLabels['not-runtime-tested']:tx('暂无实测指标','No measured metrics available')):'';
        const reports=checks.map((check,i)=>`<a class="link" href="${esc(reportLink(c,'verification',check.id,state,issueId,solution.id))}">${tx('查看复验报告','View verification report')}${checks.length>1?` ${i+1}`:''} ↗</a>`).join('');
        return `<div class="lc-solution-head"><h4 id="lc-selected-solution">${esc(title)}</h4>${reports}</div>${results}${status?`<p class="lc-solution-action" role="status">${status}</p>`:''}`;
      };
      const choices=solutionRows.length?`<div class="lc-solution-choices" role="group" aria-label="${tx('修复方案','Fix options')}">${solutionRows.map(solution=>{
        const {title,summary,outcomes}=solutionDetails.get(solution.id);
        const heading=`<span class="lc-solution-head"><strong>${esc(title)}</strong><span class="lc-verification-list">${[...outcomes].map(verificationBadge).join('')}</span></span>`;
        const content=heading+(summary?`<span class="lc-solution-summary">${esc(summary)}</span>`:'');
        return solutionRows.length>1?`<button type="button" class="lc-solution-choice" aria-controls="lc-solution-result" aria-pressed="${solution===selectedSolution}" data-pick-solution="${esc(solution.id)}">${content}</button>`:`<div class="lc-solution-choice">${content}</div>`;
      }).join('')}</div>`:'';
      const solutionSection=selectedSolution||focused||issues.length?`<section class="lc-section" data-solutions><h3>${tx('方案与结果','Fix options & results')}</h3>${selectedSolution?`${choices}<div id="lc-solution-result" class="lc-solution" role="region" aria-labelledby="lc-selected-solution" data-solution="${esc(selectedSolution.id)}">${renderSolution(selectedSolution)}</div>`:`<p class="lc-muted">${tx('暂无修复方案','No fix option recorded')}</p>`}</section>`:'';
      const shownRecords=new Set([focused?.config?.recordId,...effects.map(e=>e.record.id)].filter(Boolean));
      const shownConfigs=focused&&shownRecords.size?c.configurations.filter(config=>config.kind==='repair'&&shownRecords.has(config.recordId)):c.configurations;
      const configurations=shownConfigs.length?`<section class="lc-section lc-configurations"><h3>${tx('测试环境','Test environment')}</h3>${shownConfigs.map(config=>`<div><p>${esc(environmentLabel(config))}</p><ul>${config.libraries.map(lib=>`<li><b>${esc(lib.name)}</b>${libraryVersion(lib)?` <span>${esc(libraryVersion(lib))}</span>`:''}</li>`).join('')}</ul></div>`).join('')}</section>`:'';
      const request=`<dl class="lc-request">${[['hardware',tx('硬件','Hardware')],['model',tx('模型','Model')],['framework',tx('库 / 框架','Library / framework')],['task',tx('任务','Task')]].map(([k,label])=>`<div><dt>${label}</dt><dd>${esc((k==='task'?taskLabel(d.request[k]):d.request[k])??tx('未记录','Not recorded'))}</dd></div>`).join('')}</dl>`;
      const copy=description(focused?'issue':'case',focused?.issue||row);
      const summaries=!focused&&!solutionRows.length&&!issues.length?[...new Set(c.observations.filter(o=>o.kind==='comparison').map(({record})=>description('comparison',record).summary||record.summary).filter(text=>text&&text!==copy.summary))]:[];
      const comparisonLinks=!focused?c.observations.filter(o=>o.kind==='comparison').map(({record})=>`<li><a class="link" href="${esc(reportLink(c,'comparison',record.id,state))}">${esc(description('comparison',record).title)} ↗</a></li>`).join(''):'';
      const result=summaries.length||comparisonLinks?`<section class="lc-section"><h3>${tx('评测结果','Evaluation results')}</h3>${summaries.map(text=>`<p>${esc(text)}</p>`).join('')}${comparisonLinks?`<ul class="lc-report-links">${comparisonLinks}</ul>`:''}</section>`:'';
      const issueSection=issues.length?`<section class="lc-section"><h3><span>${tx('问题','Issues')}</span></h3>${issues.map(issue=>{
        const text=description('issue',issue),r=resolutionRows.find(row=>row.issue.id===issue.id);
        return `<div class="lc-issue">${r?resolutionBadge(r.resolution):''}<h4>${esc(text.title)}</h4>${text.summary?`<p>${esc(text.summary)}</p>`:''}</div>`;
      }).join('')}</section>`:'';
      detail.innerHTML=`<header class="lc-detail-head"><h2 id="lc-case-title" tabindex="-1">${esc(copy.title)}</h2>${focused?resolutionBadge(focused.resolution):''}</header>${copy.summary?`<p class="lc-detail-scope">${esc(copy.summary)}</p>`:''}${issueSection}${solutionSection}${result}${configurations||request}`;
      const body=detail.querySelector('#lc-solution-result');
      const refreshSolution=()=>{
        body.dataset.solution=selectedSolution.id;
        body.innerHTML=renderSolution(selectedSolution);
      };
      const buttons=[...detail.querySelectorAll('[data-pick-solution]')];
      for(const button of buttons)button.onclick=()=>{
        if(button.dataset.pickSolution===selectedSolution.id)return;
        selectedSolution=solutionRows.find(solution=>solution.id===button.dataset.pickSolution);
        for(const choice of buttons)choice.setAttribute('aria-pressed',String(choice===button));
        refreshSolution();
      };
      const finishResults=()=>{
        pending=false;
        if(body.isConnected&&dialog.open&&root._lifecycleRequest===stamp)refreshSolution();
      };
      if(body&&bindings.length)loadSolutionResults().then(async data=>{
        const definitions=data[row.id]||[];
        const all=await Promise.allSettled(definitions.map(group=>resolveResultGroup(group,allBindings,repairs,tx)));
        groups=all.flatMap(result=>result.status==='fulfilled'&&result.value?[result.value]:[]);
        failedVerifications=new Set(all.flatMap((result,i)=>result.status==='rejected'?Object.keys(definitions[i].results):[]));
        finishResults();
      }).catch(()=>{failedVerifications=new Set(bindings.map(({verification})=>verification.id));finishResults();});
    }
    function openCase(c,trigger,issueId='') {
      if(!c)return;selected=c.row.id;selectedIssue=trigger?.dataset.issue||issueId;lastTrigger=trigger||null;copyLink.textContent=tx('复制链接','Copy link');showCase(c,selectedIssue);
      const missing=configurationDifferences(c.configurations.find(config=>configurationMatches(config,state))||c.configurations.map(config=>({config,n:configurationDifferences(config,state).length})).sort((a,b)=>a.n-b.n)[0].config,state);
      if(missing.length)detail.insertAdjacentHTML('afterbegin',`<p class="lc-near-warning">${tx('配置差异','Configuration differences')}: ${missing.map(k=>esc(labels[k])).join(' · ')}</p>`);
      if(!dialog.open)dialog.showModal();updateLock();dialog.scrollTop=0;detail.querySelector('#lc-case-title').focus({preventScroll:true});writeURL();
      list.querySelectorAll('[data-case]').forEach(b=>b.setAttribute('aria-expanded',String(b.dataset.case===selected&&(!selectedIssue||b.dataset.issue===selectedIssue))));
    }
    function render() {
      refreshSelectors();
      const lookup=retrieve(index,state,{limit:1,allow_alternatives:allowAlternatives});
      const recommendation=recommendConfiguration(index,{...state,allowAlternatives});
      const issues=configurationIssues(index,recommendation?{...state,version:recommendation.library.version||''}:state);
      const ready=['hardware','model','library'].every(field=>state[field]);
      const section=root.querySelector('.lc-recommendation');
      section.innerHTML=`<h2>${tx('推荐配置','Recommended configuration')}${showingExample?` <span class="lc-badge" data-example-label>${tx('示例','Example')}</span>`:''}</h2>`+(recommendation?(()=>{
        const {case:c,config,library,solution,verification}=recommendation;
        const version=libraryVersion(library),requested=requestedVersion(state.version);
        const details=solution?'':`<button class="link lc-detail-link" type="button" data-case="${esc(c.row.id)}" aria-haspopup="dialog">${tx('配置与验证详情','Configuration & verification')} ↗</button>`;
        const report=verification||config.kind==='comparison'?`<a class="link" href="${esc(reportLink(c,verification?'verification':'comparison',verification?.id||config.recordId,state,'',solution?.id))}">${tx('查看实测报告','View measured report')} ↗</a>`:'';
        return `<div class="lc-recommend-head"><h3>${esc(library.name)}${version?` <span>${esc(version)}</span>`:''}${solution?` <span>＋ ${tx('本地修复','local fix')}</span>`:''}</h3></div>
          ${recommendation.differences.includes('version')?`<p class="lc-version-change">${version&&version!==requested?`${tx('版本调整','Version change')}: ${esc(requested)} → ${esc(version)}`:tx('已显示其他版本的记录','Showing records for a different version')}</p>`:''}
          <dl class="lc-recommend-config"><div><dt>${tx('硬件','Hardware')}</dt><dd title="${esc(config.hardware)}">${esc(state.hardware)}</dd></div><div><dt>${tx('实测模型','Tested model')}</dt><dd>${esc(config.model)}</dd></div><div><dt>${tx('任务','Task')}</dt><dd>${esc(taskLabel(state.task||config.tasks[0]||tx('未记录','Not recorded')))}</dd></div></dl>
          <section class="lc-stack" aria-label="${tx('库版本搭配','Library version combination')}"><h4>${tx('库版本搭配','Library version combination')}</h4><dl class="lc-recommend-config">${config.libraries.map(lib=>`<div><dt>${esc(lib.name)}</dt><dd>${esc(libraryVersion(lib)||'—')}</dd></div>`).join('')}</dl></section>
          ${details||report?`<div class="lc-report-links">${details}${report}</div>`:''}`;
      })():!ready?`<p class="lc-placeholder">${tx('请选择硬件、模型和框架','Select hardware, model and framework')}</p>`:
        lookup.recommendation_gaps.includes('specific-model-required-for-recommendation')?`<button class="btn sm" type="button" data-pick-model>${tx('选择具体型号','Select a specific model')}</button>`:
        `<p class="lc-placeholder">${tx('暂无可推荐组合','No supported recommendation')}</p>`);
      section.querySelector('[data-pick-model]')?.addEventListener('click',()=>openOptions('model'));
      if(!recommendation&&lookup.alternatives.length)section.insertAdjacentHTML('beforeend',`<button class="btn sm" type="button" data-show-alternatives>${tx('查看其他版本','View other versions')}</button>`);
      section.querySelector('[data-show-alternatives]')?.addEventListener('click',()=>{allowAlternatives=true;render();});
      list.innerHTML=issues.map(({issue,case:c,config,libraries,resolution})=>{
        const versions=libraries.map(name=>{const lib=config.libraries.find(lib=>lib.name===name);return [name,lib&&libraryVersion(lib)].filter(Boolean).join(' ');});
        const text=description('issue',issue);
        const note=resolutionNote(resolution);
        const meta=[config.model,...(versions.length?versions:[state.library]),note].filter(Boolean).map(esc).join(' · ');
        return `<button class="lc-problem" type="button" data-case="${esc(c.row.id)}" data-issue="${esc(issue.id)}" aria-haspopup="dialog" aria-expanded="false"><strong>${esc(text.title)}</strong><span class="lc-problem-meta">${meta}</span>${resolutionBadge(resolution)}<span class="lc-detail-link">${tx('查看详情','View details')} <span aria-hidden="true">→</span></span></button>`;
      }).join('')||`<p class="lc-placeholder">${ready?tx('暂无问题记录','No issue recorded'):tx('选择配置后查看','Select a configuration')}</p>`;
      const records=root.querySelector('[data-model-records]');
      const evaluations=state.model?filterCases(index,state).filter(c=>c.category!=='workflow-benchmark').flatMap(c=>{
        const matches=new Map();
        for(const config of c.configurations){
          if(config.kind==='comparison'&&configurationMatches(config,state))matches.set(config.recordId,config);
        }
        return [...matches.values()].map(config=>({c,config,record:get('comparison',config.recordId)}));
      }).sort((a,b)=>String(b.record.date).localeCompare(String(a.record.date))||a.record.id.localeCompare(b.record.id)):[];
      records.hidden=!evaluations.length;
      const evaluationList=items=>items.length?`<ul class="lc-evaluation-links">${items.map(({c,config,record})=>{
        const size=config.searchConfig.model_size;
        return `<li><button class="link" type="button" data-case="${esc(c.row.id)}" aria-haspopup="dialog">${esc(description('comparison',record).title)}</button><span>${esc(config.model)}${['tiny','unknown'].includes(size)||choiceKey(config.model).includes(choiceKey(size))?'':' · '+esc(display('model_size',size))}</span><a class="link" href="${esc(reportLink(c,'comparison',record.id,state))}">${tx('实测报告','Measured report')} ↗</a></li>`;
      }).join('')}</ul>`:'';
      const tiny=evaluations.filter(({config})=>config.searchConfig.model_size==='tiny');
      records.innerHTML=evaluations.length?`<h2>${tx('评测记录','Evaluations')} · ${evaluations.length}</h2>${evaluationList(evaluations.filter(({config})=>config.searchConfig.model_size!=='tiny'))}${tiny.length?`<details class="lc-option-group" ${tiny.length===evaluations.length?'open':''}><summary>${display('model_size','tiny')} · ${tiny.length}</summary>${evaluationList(tiny)}</details>`:''}`:'';
      root.querySelectorAll('[data-case]').forEach(button=>button.onclick=()=>openCase(index.cases.find(c=>c.row.id===button.dataset.case),button));
      writeURL();
    }
    root.querySelector('[data-reset]').onclick=()=>{for(const k of configurationFields)state[k]='';state.q='';allowAlternatives=false;showingExample=false;selected='';render();};
    const requested=selected,requestedIssue=selectedIssue;selected='';selectedIssue='';render();
    if(requested){const c=index.cases.find(c=>c.row.id===requested);if(c)openCase(c,null,requestedIssue);}
  }

  async function mountReport(root,hash,{lang='zh',copyLink}={}) {
    const tx=(zh,en)=>lang==='en'?en:zh;
    const request={};root._reportRequest=request;
    const active=()=>root._reportRequest===request&&!root.hidden;
    const backParams=new URLSearchParams();
    for(const field of [...configurationFields,'issue','solution'])if(hash.q.get(field))backParams.set(field,hash.q.get(field));
    const back=hash.arg?`#/cases/${encodeURIComponent(hash.arg)}${backParams.size?'?'+backParams:''}`:'#/cases';
    const backLink=`<a class="link" href="${esc(back)}">← ${tx('返回案例','Back to case')}</a>`;
    const paint=html=>{if(active())root.innerHTML=`<article class="wrap lc-report">${backLink}${html}</article>`;};
    const noReport=()=>paint(`<h1>${tx('未找到实测报告','Measured report not found')}</h1>`);
    paint(`<p role="status">${tx('正在读取实测报告…','Loading measured report…')}</p>`);
    let index;
    try{index=await loadIndex();}catch{if(active())paint(`<h1>${tx('无法读取实测报告','Unable to load the measured report')}</h1>`);return;}
    if(!active())return;
    const c=index.cases.find(({row})=>row.id===hash.arg);
    const comparisonId=hash.q.get('comparison'),verificationId=hash.q.get('verification');
    if(!c||Boolean(comparisonId)===Boolean(verificationId))return noReport();
    const comparison=comparisonId&&c.observations.find(o=>o.kind==='comparison'&&o.record.id===comparisonId)?.record;
    const bindings=verificationId?verificationBindings(index,c):[];
    const binding=bindings.find(b=>b.verification.id===verificationId&&(!hash.q.get('solution')||b.solution.id===hash.q.get('solution')));
    if(!comparison&&!binding)return noReport();

    const describe=(kind,row)=>Search.describe(index,kind,row.id,lang,{title:row.details?.title||row.title,summary:row.summary});
    const section=(title,content)=>content?`<section class="lc-report-section"><h2>${title}</h2>${content}</section>`:'';
    const list=rows=>rows?.length?`<ul>${rows.map(text=>`<li>${esc(text)}</li>`).join('')}</ul>`:'';
    const text=value=>value?`<p>${esc(value)}</p>`:'';
    const format=value=>typeof value==='number'&&Number.isFinite(value)?String(value):value??'—';
    const coverage=rows=>list((rows||[]).filter(row=>row&&row.observed!=null).map(row=>
      `${row.label||tx('覆盖','Coverage')}: ${format(row.observed)}${row.expected!=null?' / '+format(row.expected):''}${row.unit?' '+row.unit:''}`));
    const table=(head,rows)=>rows.length?`<table class="lc-report-table"><thead><tr>${head.map(v=>`<th scope="col">${esc(v)}</th>`).join('')}</tr></thead><tbody>${rows.map(([first,...values])=>`<tr><th scope="row">${esc(first)}</th>${values.map(v=>`<td>${esc(format(v))}</td>`).join('')}</tr>`).join('')}</tbody></table>`:'';
    const versions=side=>table([tx('组件','Component'),tx('实测版本','Recorded version')],(side.libraries||[]).map(({name,version,commit})=>{
      const {number,kind}=versionParts({version,commit});
      return [name,number?(kind==='source'?number+' · '+tx('源码','Source'):number):kind==='source'?tx('源码版本','Source build'):version||'—'];
    }));
    const settings=(value,prefix='')=>Object.entries(value||{}).flatMap(([name,v])=>{
      if(/sha\d*|checksum|hash|revision|commit|(?:^|_)path(?:$|_)/i.test(name))return [];
      const label=prefix?`${prefix} / ${name}`:name;
      if(v&&typeof v==='object'&&!Array.isArray(v))return settings(v,label);
      if(Array.isArray(v))return v.every(item=>['number','string','boolean'].includes(typeof item))?[[label,v.join(', ')]]:[];
      return v==null?[]:[[label,String(v)]];
    });
    const sideCard=side=>{
      const rows=settings(side.settings);
      return `<div class="lc-report-side">${side.label?`<h3>${esc(side.label)}</h3>`:''}${text(side.hardware)}${versions(side)}${rows.length?`<details><summary>${tx('运行参数','Runtime settings')}</summary>${table([tx('参数','Setting'),tx('值','Value')],rows)}</details>`:''}</div>`;
    };
    const sourceFiles=rows=>[...new Set(rows.filter(path=>typeof path==='string'&&/^evidence\/(?:[\w.-]+\/)*[\w.-]+\.(?:md|json|csv)$/i.test(path)&&!path.split('/').some(p=>p==='..'||p==='.')))];
    const sourceLinks=files=>files.length?`<ul>${files.map((path,i)=>`<li><a class="link" href="./${esc(path)}">${path.endsWith('.md')?tx('完整报告','Full report'):tx('原始记录','Original record')}${files.length>1?' '+(i+1):''} ↗</a></li>`).join('')}</ul>`:'';
    const taskLabel=task=>({inference:tx('推理','Inference'),sft:'SFT',rl:'RL'}[task]||task);
    let title,meta,body,exportData;
    if(comparison){
      const {sides,comparisons,common,differences,notes}=comparison;
      const copy=describe('comparison',comparison);
      title=copy.title;meta=[comparison.model?.name,taskLabel(comparison.task),comparison.date];
      const sideNames=new Map(sides.map(side=>[side.id,side.label||side.hardware]));
      const results=comparisons.map(record=>{
        const sideColumns=(record.side_ids||[]).filter(id=>(record.metrics||[]).some(m=>Object.hasOwn(m.side_values||{},id)));
        return `<div class="lc-report-item"><h3>${esc(record.title)}</h3>${text((record.side_ids||[]).map(id=>sideNames.get(id)||id).join(' ↔ '))}${text(record.scope)}${coverage(record.coverage)}${table([tx('指标','Metric'),tx('结果','Result'),tx('单位','Unit'),...sideColumns.map(id=>sideNames.get(id)||id)],(record.metrics||[]).map(m=>[m.label||m.id,m.value,m.unit,...sideColumns.map(id=>m.side_values?.[id])]))}</div>`;
      }).join('');
      body=section(tx('对照与范围','Comparison scope'),text(copy.summary)+list([...(common||[]),...(differences||[])]))+
        section(tx('实测环境','Test environment'),`<div class="lc-report-sides">${sides.map(sideCard).join('')}</div>`)+
        section(tx('实测指标','Measured results'),results)+section(tx('补充说明','Notes'),list(notes))+
        section(tx('来源','Sources'),sourceLinks(sourceFiles((comparison.evidence||[]).map(e=>e.url))));
      exportData={kind:'comparison',case_id:c.row.id,record:comparison};
    }else{
      const {verification,solution}=binding,repairs=caseRepairs(c),details=verification.details;
      const refs=details.sources;
      const workloads=index.relations.links.filter(l=>l.relation==='uses_workload'&&l.from.kind==='verification'&&l.from.id===verification.id&&l.to.kind==='workload'&&c.entries.some(e=>e.kind==='workload'&&e.id===l.to.id))
        .map(l=>index.records.get(key('workload',l.to.id)));
      const workloadRefs=workloads.flatMap(w=>w.details.sources);
      const explicitRepairs=(sources,pointer)=>[...new Set(sources.filter(s=>s.kind==='repair'&&s.pointer===pointer).map(s=>repairs.get(s.id)).filter(Boolean))];
      const environments=explicitRepairs(workloadRefs,'/environment');
      const metricRepairs=explicitRepairs(refs,'/validation/metrics');
      const checkRepairs=explicitRepairs(refs,'/validation/checks');
      const copy=describe('candidate',solution);
      title=copy.title;
      meta=[...new Set(environments.map(r=>r.model?.name)),...new Set(environments.map(r=>taskLabel(r.task)))];
      let groups=[],unavailable=false;
      try{
        const definitions=(await loadSolutionResults())[c.row.id]||[];
        const outcomes=await Promise.allSettled(definitions.filter(g=>Object.hasOwn(g.results,verification.id)).map(g=>resolveResultGroup(g,bindings,repairs,tx)));
        unavailable=outcomes.some(x=>x.status==='rejected');
        groups=outcomes.flatMap(x=>x.status==='fulfilled'&&x.value?x.value.columns.filter(col=>col.verification.id===verification.id&&col.solution.id===solution.id).map(col=>({label:x.value.label,metrics:col.metrics})):[]);
      }catch{unavailable=true;}
      if(!groups.length)groups=metricRepairs.map(r=>({label:r.validation.scope,metrics:r.validation.metrics.filter(m=>Number.isFinite(m.before)&&Number.isFinite(m.after))})).filter(g=>g.metrics.length);
      const checks=checkRepairs.flatMap(r=>r.validation.checks||[]);
      const covered=explicitRepairs(refs,'/validation/coverage').map(r=>r.validation.coverage).filter(Boolean);
      const workloadCoverage=explicitRepairs(workloadRefs,'/validation/coverage').map(r=>r.validation.coverage).filter(Boolean);
      const reference=metricRepairs.map(r=>r.problem?.expected).filter(Boolean);
      const limits=details.limitations||[];
      const outcome={ 'scoped-checks-passed':tx('检查通过','Checks passed'),partial:tx('部分检查通过','Some checks passed'),
        'mixed-results':tx('指标有取舍','Mixed results'),'not-adopted':tx('未采用','Not adopted') }[details.outcome]||details.outcome;
      body=section(tx('复验范围','Verification scope'),text(outcome)+text(copy.summary)+list([...new Set([verification.scope,...workloads.map(w=>w.scope),...reference].filter(Boolean))])+coverage(covered)+
        (workloadCoverage.length?coverage(workloadCoverage.map(row=>({...row,label:tx('负载覆盖','Workload coverage')}))):''))+
        section(tx('实测环境','Test environment'),environments.length?`<div class="lc-report-sides">${environments.map(r=>sideCard(r.environment)).join('')}</div>`:'')+
        section(tx('实测指标','Measured results'),groups.map(({label,metrics})=>`<div class="lc-report-item">${text(label)}${table([tx('指标','Metric'),tx('基线','Baseline'),tx('本方案','This option'),tx('单位','Unit')],metrics.map(m=>[m.label,m.before,m.after,m.unit]))}</div>`).join('')+(unavailable?text(tx('部分实测指标暂不可读取','Some measured metrics are unavailable')):groups.length?'':text(tx('这次复验没有记录可展示的数值指标','No numerical metrics recorded for this verification'))))+
        section(tx('检查记录','Checks'),list(checks))+section(tx('未覆盖范围','Limits'),list(limits))+
        section(tx('来源','Sources'),sourceLinks(sourceFiles(refs.map(s=>s.file).concat(verification.evidence?.filter(s=>s.file.endsWith('.md')).map(s=>s.file)||[]))));
      exportData={kind:'verification',case_id:c.row.id,solution_id:solution.id,verification_id:verification.id,outcome:details.outcome,
        scope:verification.scope,workloads:workloads.map(w=>w.scope),environments:environments.map(r=>r.environment),results:groups,checks,coverage:covered,workload_coverage:workloadCoverage,limitations:limits};
    }
    if(!active())return;
    paint(`<header class="lc-report-head"><p class="eyebrow">${tx('实测报告','Measured Report')}</p><h1>${esc(title)}</h1><p>${meta.filter(Boolean).map(esc).join(' · ')}</p><div class="lc-report-actions"><button class="btn sm" type="button" data-report-copy>${tx('复制链接','Copy link')}</button><button class="btn sm" type="button" data-report-export>${tx('导出 JSON','Export JSON')}</button></div></header>${body}`);
    root.querySelector('[data-report-copy]').onclick=e=>copyLink?.(global.location.href,e.currentTarget);
    root.querySelector('[data-report-export]').onclick=()=>{
      const url=URL.createObjectURL(new Blob([JSON.stringify(exportData,null,2)+'\n'],{type:'application/json'}));
      const link=document.createElement('a');link.href=url;link.download='numericbridge-report.json';link.click();
      setTimeout(()=>URL.revokeObjectURL(url),1000);
    };
  }

  global.NumericBridgeCases={buildIndex,filterCases,configurationDifferences,configurationMatches,assessCoverage,recommendConfiguration,configurationIssues,issueResolution,issueSolutions,mount,mountReport};
  if(typeof module!=='undefined'&&module.exports)module.exports=global.NumericBridgeCases;
})(typeof window!=='undefined'?window:globalThis);
