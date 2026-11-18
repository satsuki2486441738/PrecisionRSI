/* Query the same compact metadata used by the standalone Python tool.
 * Scientific recommendation eligibility is compiled once by community.records.search.
 */
(function(global){
  'use strict';
  const fields=['hardware','model','library','task','version','model_size'],core=['hardware','model','library'];
  const clone=value=>JSON.parse(JSON.stringify(value));
  const key=value=>String(value).trim().toLowerCase().replace(/\s+/g,' ');
  const cmp=(a,b)=>a<b?-1:a>b?1:0;
  function describe(index,kind,id,language='zh',fallback={}){
    const content=index?.content?.[kind+':'+id]||{};
    const text=field=>content[field]?.[language]||content[field]?.en||content[field]?.zh||fallback[field]||'';
    return {title:text('title'),summary:text('summary')};
  }
  function hardwarePlatforms(value,index){
    const text=String(value).trim();
    const hardware=text.toLowerCase().replace(/\bno\s+(?:model\s+or\s+)?(?:accelerators?|gpus?|npus?)\b/g,' ');
    let names=[...new Set(index.hardware_rules.filter(r=>new RegExp(r.pattern,'u').test(hardware)).map(r=>r.name))];
    if(names.some(n=>['Ascend 910C','Ascend 910B'].includes(n)))names=names.filter(n=>n!=='Ascend NPU');
    if(names.some(n=>/^(NVIDIA H|RTX )/.test(n)))names=names.filter(n=>n!=='NVIDIA GPU');
    if(names.some(n=>n!=='CPU'))names=names.filter(n=>n!=='CPU');
    if(names.length===1&&names[0]==='NVIDIA GPU'&&/(?:nvidia|geforce)\s*(?:[a-z]+\s*)?\d/.test(hardware))return [text];
    if(names.length===1&&names[0]==='CPU'&&/offload|gpu|npu|accelerator/.test(hardware))return [text];
    return names.length?names:[text];
  }
  function normalizeRequest(request,index){
    if(!request||typeof request!=='object'||Array.isArray(request)||Object.keys(request).some(f=>![...fields,'query'].includes(f)))throw new Error('Unsupported search request fields');
    const values={};
    for(const field of [...fields,'query']){
      if(request[field]!=null&&typeof request[field]!=='string')throw new Error(field+' must be text or null');
      const value=(request[field]||'').trim();if(value.length>1000)throw new Error(field+' is too long');
      values[field]=value?(index.aliases[field]?.[key(value)]||value):'';
    }
    if(values.hardware){const platforms=hardwarePlatforms(values.hardware,index);if(platforms.length!==1)throw new Error('Choose one target hardware platform');values.hardware=platforms[0];}
    if(values.version&&!values.library)throw new Error('version requires library');
    if(values.model_size&&!values.model)throw new Error('model_size requires model');
    if(values.task&&!['inference','sft','rl'].includes(values.task))throw new Error('Unsupported task');
    return values;
  }
  function differences(config,request,ignoreVersion=false){
    return fields.filter(field=>{
      const value=request[field];if(!value||ignoreVersion&&field==='version')return false;
      if(field==='version')return !config.libraries.some(lib=>lib.keys.includes(key(request.library))&&lib.version===value);
      if(field==='task')return !config.tasks.includes(value);
      if(field==='model_size')return key(config.model_size||'')!==key(value);
      return !config.keys[field].includes(key(value));
    });
  }
  function configuration(config){
    const result=clone(Object.fromEntries(Object.entries(config).filter(([k])=>!['keys','recommendations'].includes(k))));
    result.libraries=result.libraries.map(lib=>Object.fromEntries(Object.entries(lib).filter(([k])=>k!=='keys')));return result;
  }
  function caseResult(row,configs,diff=[]){
    const recordIds=new Set(configs.map(c=>c.record.id));
    return {id:row.id,title:row.title,category:row.category,scope:row.scope,content:clone(row.content||{}),source:clone(row.source),evidence:clone(row.evidence),
      configurations:configs.map(configuration),issues:clone(row.issues.filter(i=>i.record_ids.some(id=>recordIds.has(id)))),
      solutions:clone(row.solutions),differences:diff};
  }
  function search(index,request,options={}){
    if(index?.kind!=='numericbridge-search-index'||index.schema_version!==1||index.policy_version!==2)throw new Error('Unsupported search metadata');
    const {limit=20,offset=0,related_limit=0,include_workflows=false,allow_alternatives=false}=options;
    for(const [value,min,max,name] of [[limit,1,100,'limit'],[offset,0,100000,'offset'],[related_limit,0,20,'related_limit']])if(!Number.isInteger(value)||value<min||value>max)throw new Error('Invalid '+name);
    const normalized=normalizeRequest(request,index),missing=core.filter(f=>!normalized[f]),terms=key(normalized.query).split(' ').filter(Boolean);
    const matches=[],related=[],choices=[],alternatives=[],gaps=new Set();let configurationCount=0;
    for(const row of [...index.cases].sort((a,b)=>cmp(a.id,b.id))){
      if(row.category==='workflow-benchmark'&&!include_workflows)continue;
      const exact=row.configurations.filter(c=>!differences(c,normalized).length);if(exact.length)configurationCount++;
      if(!terms.every(term=>row.search_text.includes(term)))continue;
      if(exact.length){matches.push(caseResult(row,exact));exact.flatMap(c=>c.recommendation_gaps).forEach(g=>gaps.add(g));}
      if(!exact.length&&related_limit){
        const ranked=row.configurations.map(c=>({config:c,diff:differences(c,normalized)})).filter(r=>r.diff.length).sort((a,b)=>a.diff.length-b.diff.length||cmp(a.config.id,b.config.id));
        if(ranked.length&&ranked[0].diff.length<fields.filter(f=>normalized[f]).length&&core.filter(f=>ranked[0].diff.includes(f)).length<=1)related.push({n:ranked[0].diff.length,id:row.id,result:caseResult(row,[ranked[0].config],ranked[0].diff)});
      }
      if(missing.length||row.category==='workflow-benchmark')continue;
      for(const config of row.configurations){
        if(differences(config,normalized,true).length)continue;
        if(!(config.keys.model_identity??[key(config.model)]).includes(key(normalized.model))){
          if(config.recommendations.length)gaps.add('specific-model-required-for-recommendation');
          continue;
        }
        for(const support of config.recommendations){
          if(support.library_keys?.length&&!support.library_keys.includes(key(normalized.library)))continue;
          const option={case_id:row.id,configuration:configuration(config),...clone(Object.fromEntries(Object.entries(support).filter(([k])=>k!=='library_keys')))};
          option.differences=differences(config,normalized);option.version_match=option.differences.includes('version')?'alternative':'exact';
          (option.differences.length?alternatives:choices).push(option);
        }
      }
    }
    const order=o=>[o.candidate_id?0:1,-Number((o.configuration.date||'').replace(/-/g,'')||0),o.configuration.record.id,o.configuration.id,o.candidate_id||'',o.case_id];
    const compare=(a,b)=>{const aa=order(a),bb=order(b);for(let i=0;i<aa.length;i++){const c=cmp(aa[i],bb[i]);if(c)return c;}return 0;};
    const distinct=rows=>{const found=new Map();for(const row of rows.sort(compare)){const id=JSON.stringify([row.configuration.id,row.candidate_id]);if(!found.has(id))found.set(id,row);}return [...found.values()];};
    const exactChoices=distinct(choices),otherChoices=distinct(alternatives),recommendations=exactChoices.length?exactChoices:allow_alternatives?otherChoices:[];
    if(missing.length)gaps.add('missing-required-input');else if(!configurationCount)gaps.add('no-matching-configuration');else if(!recommendations.length)gaps.add('insufficient-recommendation-evidence');
    return {schema_version:1,normalized_request:normalized,missing_input:missing,
      status:missing.length?'needs-input':matches.length?'matched':configurationCount?'no-keyword-match':related.length?'related-only':'no-records',
      snapshot:{sources:clone(index.sources),policy_version:index.policy_version,observation_date:index.observation_date},
      configuration_case_count:configurationCount,total:matches.length,offset,limit,matches:matches.slice(offset,offset+limit),
      related:matches.length?[]:related.sort((a,b)=>a.n-b.n||cmp(a.id,b.id)).slice(0,related_limit).map(r=>r.result),
      recommendation_status:recommendations.length?'available':missing.length?'needs-input':'insufficient-evidence',recommendation:recommendations[0]||null,
      recommendations:recommendations.slice(0,10),alternatives:otherChoices.slice(0,10),recommendation_gaps:recommendations.length?[]:[...gaps].sort()};
  }
  global.NumericBridgeSearch={search,normalizeRequest,hardwarePlatforms,configurationDifferences:differences,describe};
  if(typeof module!=='undefined'&&module.exports)module.exports=global.NumericBridgeSearch;
})(typeof window!=='undefined'?window:globalThis);
