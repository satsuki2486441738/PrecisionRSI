#!/usr/bin/env python3
"""Replay18 anonymous I04 target pairs,159/95 separately; open0 native arrays.
Usage: python -S -B replay-targets.py selected-targets.json [selected-metrics.json]
Distribution scalars and E02/cost summaries are archived, not recomputed here.
"""
import hashlib,json,math
from pathlib import Path
import sys
TARGET_SHA='39d15fa7c8312c6783ce02dfda039352dd36f58e8c790fb21ace5f9c9b14bf61'
METRIC_SHA='7f9b478c857d0e7002d1893338d2e37b15fc6b9a3bfb770ea88b060cc5e4e61b'
PAIRS=[(['gpu-alone-a-a', 'npu-alone-a-a'], 'A', 'cross_backend_matched'), (['gpu-batch-left-a', 'npu-batch-left-a'], 'A', 'cross_backend_matched'), (['gpu-batch-right-a', 'npu-batch-right-a'], 'A', 'cross_backend_matched'), (['gpu-alone-b-b', 'npu-alone-b-b'], 'B', 'cross_backend_matched'), (['gpu-batch-left-b', 'npu-batch-left-b'], 'B', 'cross_backend_matched'), (['gpu-batch-right-b', 'npu-batch-right-b'], 'B', 'cross_backend_matched'), (['gpu-alone-a-a', 'gpu-batch-left-a'], 'A', 'within_backend_batch_padding'), (['gpu-alone-a-a', 'gpu-batch-right-a'], 'A', 'within_backend_batch_padding'), (['gpu-batch-left-a', 'gpu-batch-right-a'], 'A', 'within_backend_batch_padding'), (['gpu-alone-b-b', 'gpu-batch-left-b'], 'B', 'within_backend_batch_padding'), (['gpu-alone-b-b', 'gpu-batch-right-b'], 'B', 'within_backend_batch_padding'), (['gpu-batch-left-b', 'gpu-batch-right-b'], 'B', 'within_backend_batch_padding'), (['npu-alone-a-a', 'npu-batch-left-a'], 'A', 'within_backend_batch_padding'), (['npu-alone-a-a', 'npu-batch-right-a'], 'A', 'within_backend_batch_padding'), (['npu-batch-left-a', 'npu-batch-right-a'], 'A', 'within_backend_batch_padding'), (['npu-alone-b-b', 'npu-batch-left-b'], 'B', 'within_backend_batch_padding'), (['npu-alone-b-b', 'npu-batch-right-b'], 'B', 'within_backend_batch_padding'), (['npu-batch-left-b', 'npu-batch-right-b'], 'B', 'within_backend_batch_padding')]
def require(ok,message):
 if not ok:raise ValueError(message)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def load(path,digest):
 p=Path(path);require(p.is_file() and not p.is_symlink() and p.stat().st_size<=16*1024**2,'bounded public JSON required')
 raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==digest,'frozen public artifact SHA differs')
 def pairs(items):
  out={}
  for k,v in items:require(k not in out,'duplicate JSON key');out[k]=v
  return out
 def reject(v):raise ValueError('nonfinite JSON token '+v)
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=reject)
def calculate(a,b):
 n=len(a);require(n in (159,95) and len(b)==n and all(type(x) in (int,float) and math.isfinite(x) and x<=0 for x in a+b),'complete159/95 vectors required')
 e=[abs(x-y) for x,y in zip(a,b)];s=sorted(e);h=.99*(n-1);lo=math.floor(h);hi=math.ceil(h);na,nb=[math.fsum(-v for v in x)/n for x in (a,b)]
 return {'target_count':n,'side_a_nll_nat_per_token':na,'side_b_nll_nat_per_token':nb,'nll_abs_diff_nat_per_token':abs(na-nb),'mae_nat':math.fsum(e)/n,'p99_nat':s[lo]+(h-lo)*(s[hi]-s[lo]),'p99_method':'global linear h=0.99*(N-1)','max_abs_nat':max(e),'max_target_index_zero_based':e.index(max(e))}
def replay(target_path,metric_path=None):
 p=Path(target_path);t=load(p,TARGET_SHA);m=load(Path(metric_path) if metric_path else p.with_name('selected-metrics.json'),METRIC_SHA)
 require(t['kind']=='anonymous_i04_target_replay' and m['kind']=='anonymous_i04_comparison_metrics','I04 public artifact kinds required')
 for d in (t,m):require(d['thresholds'] is None and d['acceptance'] is None and d['full_core_complete'] is False and d['workload']['independent_parent_inputs']==1 and d['workload']['target_distribution_pairs']==18 and d['array_evidence']['public_target_replay_arrays_read']==0,'measured partial scope differs')
 for k in ('model','source_pins','workload','pair_scope','array_evidence','target_pairs'):require(t[k]==m[k],'public artifacts disagree: '+k)
 require(m['target_replay_artifact']=={'file':'selected-targets.json','sha256':TARGET_SHA},'target artifact binding differs')
 require([(p['sides'],p['request'],p['kind']) for p in t['target_pairs']]==PAIRS,'exact18 cross-first pair inventory required')
 vectors={}
 for i,r in enumerate(t['runs']):
  values=r['target_logprobs_nat'];n=159 if r['request']=='A' else 95
  require(r['target_count']==len(values)==n and hashlib.sha256(canonical(values).encode()).hexdigest()==r['target_vector_sha256'],'anonymous vector identity differs')
  require({k:v for k,v in r.items() if k!='target_logprobs_nat'}==m['runs'][i],'run metadata differs');vectors[r['id']]=values
 require(len(vectors)==12 and len(m['runs'])==12,'12 distinct logical observations required')
 for i,p in enumerate(t['target_pairs']):
  a,b=p['sides'];n=159 if p['request']=='A' else 95;require(p['coverage']=={'expected':n,'paired':n},'per-request coverage differs')
  actual=calculate(vectors[a],vectors[b]);require(set(actual)==set(p['metrics']),'complete target scalar inventory required')
  for k,v in actual.items():
   old=p['metrics'][k]
   if type(v) in (int,str):require(type(old) is type(v) and old==v,'target discrete scalar differs')
   else:require(type(old) in (int,float) and math.isfinite(old) and math.isclose(v,old,rel_tol=0,abs_tol=1e-12),'target scalar differs')
  dist=m['distribution_pairs'][i];require(dist['sides']==[a,b] and dist['request']==p['request'] and dist['position_count']==len(dist['positions'])==47,'archived distribution scope differs')
 return {'status':'recorded_i04_target_metrics_replayed','target_pairs_checked':18,'request_target_counts':{'A':159,'B':95},'independent_parent_inputs':1,'derived_prefix_shape_stress_requests':1,'target_metric_verification_tolerance':1e-12,'verification_tolerance_is_model_acceptance_threshold':False,'native_executions_started':0,'arrays_read':0,'archived_distribution_metrics_replayed':False,'thresholds':None,'acceptance':None,'full_core_complete':False}
if __name__=='__main__':
 if len(sys.argv) not in (2,3):raise SystemExit('usage: python -S -B replay-targets.py selected-targets.json [selected-metrics.json]')
 try:result=replay(sys.argv[1],sys.argv[2] if len(sys.argv)==3 else None)
 except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:print(json.dumps({'status':'rejected','reason':str(exc),'acceptance':None}));raise SystemExit(2)
 print(json.dumps(result,indent=2,allow_nan=False))
