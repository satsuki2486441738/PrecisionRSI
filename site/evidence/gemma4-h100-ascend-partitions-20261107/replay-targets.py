#!/usr/bin/env python3
"""Replay23 anonymous I03 target comparisons with the Python stdlib.

Usage: python -S -B replay-targets.py selected-targets.json [selected-metrics.json]
Archived distribution and native KV scalars are integrity checked only.
This program opens exactly the two frozen public JSON files; no native arrays.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

TARGETS_SHA256='2d7e6c2e872c69dc49216211bb7e72eb356b79a63744437aa4278b5fe8707289'
METRICS_SHA256='bc4f3cb65d071c6aede6c7e9bcfa8489303e99091397a0793886d239497ff077'
MODES=['full-r1','full-r2','full-r3','token','chunk63-65']
SIDES=[backend+'-'+mode for backend in ('gpu','npu') for mode in MODES]
PAIRS=[['gpu-full-r%d'%i,'npu-full-r%d'%j] for i in (1,2,3) for j in (1,2,3)]+[['gpu-token','npu-token'],['gpu-chunk63-65','npu-chunk63-65']]
for backend in ('gpu','npu'):
    PAIRS.extend([[backend+'-full-r%d'%i,backend+'-full-r%d'%j] for i,j in ((1,2),(1,3),(2,3))])
    PAIRS.extend([[backend+'-full-r1',backend+'-token'],[backend+'-full-r1',backend+'-chunk63-65'],[backend+'-token',backend+'-chunk63-65']])
KV_PAIRS=[['gpu-token','npu-token'],['gpu-chunk63-65','npu-chunk63-65'],['gpu-token','gpu-chunk63-65'],['npu-token','npu-chunk63-65']]
POSITIONS=[0,8,16,25,33,42,50,59,67,76,84,93,101,110,118]+list(range(127,159))
ATOL=1e-12


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def load_frozen(path,digest):
    path=Path(path);require(path.is_file() and path.stat().st_size<=8*1024**2,'bounded frozen JSON required')
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==digest,'frozen public artifact SHA differs')
    def pairs(items):
        out={}
        for key,value in items:require(key not in out,'duplicate JSON key');out[key]=value
        return out
    def reject(value):raise ValueError('nonfinite JSON token '+value)
    doc=json.loads(raw,object_pairs_hook=pairs,parse_constant=reject);canonical(doc);return doc


def calculate(a,b):
    require(len(a)==len(b)==159,'paired target count differs')
    errors=[abs(x-y) for x,y in zip(a,b)];ordered=sorted(errors);offset=.99*(len(errors)-1)
    lo,hi=math.floor(offset),math.ceil(offset);na,nb=[math.fsum(-v for v in values)/len(values) for values in (a,b)]
    maximum=max(errors)
    return {'target_count':159,'side_a_nll_nat_per_token':na,'side_b_nll_nat_per_token':nb,'nll_abs_diff_nat_per_token':abs(na-nb),
        'mae_nat':math.fsum(errors)/159,'p99_nat':ordered[lo]+(offset-lo)*(ordered[hi]-ordered[lo]),
        'p99_method':'global linear h=0.99*(N-1)','max_abs_nat':maximum,'max_target_index_zero_based':errors.index(maximum)}


def replay(target_path,metric_path=None):
    target_path=Path(target_path);metric_path=Path(metric_path) if metric_path else target_path.with_name('selected-metrics.json')
    targets=load_frozen(target_path,TARGETS_SHA256);metrics=load_frozen(metric_path,METRICS_SHA256)
    require(targets['kind']=='anonymous_i03_target_replay' and metrics['kind']=='anonymous_i03_comparison_metrics','unsupported evidence kind')
    for doc in (targets,metrics):
        require(doc['schema_version']==1 and doc['status']=='scoped_measured_fixed_input_comparison' and doc['thresholds'] is None and
            doc['acceptance'] is None and doc['full_core_complete'] is False,'unexpected completion/acceptance claim')
        require(doc['model']=={'repository':'google/gemma-4-12B','variant':'base','revision':'1a2c7729392586f5ecddae8807ee2ef741eabb9a',
            'parameter_dtype':'BF16','attention':'eager'},'fixed model differs')
        w=doc['workload'];require(w['independent_inputs']==1 and w['new_model_processes_per_backend']==1 and w['native_calls_per_backend']==70 and
            w['modes_per_backend']==5 and w['full_repeat_calls_per_backend']==3 and w['targets_per_mode']==159 and w['distribution_rows_per_mode']==47 and
            w['native_logits_shape_per_mode']==[1,160,262144] and w['full_vocabulary_rows_rederived']==470 and w['target_scores_rederived']==1590 and
            w['raw_native_predictor_rows_finite_checked']==1600,'measured workload coverage differs')
        require(w['fresh_process_repeatability_measured'] is False and w['cached_mode_repeatability_measured'] is False and w['cache_eviction_tested'] is False,
            'unmeasured scope differs')
        require(doc['pair_scope']=={'cross_backend_full_pairs':9,'cross_backend_cached_pairs':2,'within_process_full_repeat_pairs':6,
            'within_process_partition_pairs':6,'total_target_distribution_pairs':23,'native_final_KV_pairs':4},'23/4 pair scope differs')
        require(doc['array_evidence']['public_target_replay_arrays_read']==0 and doc['array_evidence']['public_target_replay_recomputes_distribution_metrics'] is False and
            doc['array_evidence']['public_target_replay_recomputes_native_KV_metrics'] is False,'target-only replay scope differs')
    for key in ('model','source_pins','workload','pair_scope','coordinate_convention','array_evidence','target_pairs'):
        require(targets[key]==metrics[key],'public artifacts disagree: '+key)
    require(metrics['target_replay_artifact']=={'file':'selected-targets.json','sha256':TARGETS_SHA256},'target artifact binding differs')
    require([run['id'] for run in targets['runs']]==SIDES and [run['id'] for run in metrics['runs']]==SIDES,'ten ordered anonymous vectors required')
    vectors={}
    for index,run in enumerate(targets['runs']):
        values=run['target_logprobs_nat'];require(run['target_start_index_zero_based']==0 and run['target_count']==159 and isinstance(values,list) and len(values)==159 and
            all(type(v) in (int,float) and math.isfinite(v) and v<=0 for v in values),'target vector domain/values differ')
        require(hashlib.sha256(canonical(values).encode()).hexdigest()==run['target_vector_sha256'],'anonymous vector SHA differs')
        require({k:v for k,v in run.items() if k!='target_logprobs_nat'}==metrics['runs'][index],'run metadata differs')
        vectors[run['id']]=values
    require([p['sides'] for p in targets['target_pairs']]==PAIRS and [p['sides'] for p in metrics['distribution_pairs']]==PAIRS,'ordered23 pair identities differ')
    checked=[]
    for index,pair in enumerate(targets['target_pairs']):
        a,b=pair['sides'];require(pair['target_start_index_zero_based']==0 and pair['coverage']=={'expected':159,'paired':159} and
            pair['thresholds'] is None and pair['acceptance'] is None,'159target scope differs')
        actual=calculate(vectors[a],vectors[b]);require(set(actual)==set(pair['metrics']),'target metric field inventory differs')
        for key,value in actual.items():
            reference=pair['metrics'][key]
            if type(value) in (int,str):require(type(reference) is type(value) and reference==value,'discrete target metric differs: '+key)
            else:require(type(reference) in (int,float) and math.isfinite(reference) and math.isclose(value,reference,rel_tol=0,abs_tol=ATOL),'target scalar differs: '+key)
        d=metrics['distribution_pairs'][index];require(d['sides']==[a,b] and d['position_count']==47 and
            [row['prediction_index_zero_based'] for row in d['positions']]==POSITIONS and
            [row['sampled_position_index_zero_based'] for row in d['positions']]==list(range(47)),'archived distribution coordinate coverage differs')
        require(d['aggregate']['top1_agreement']['total']==47,'archived top1 coverage differs')
        checked.append({'sides':[a,b],'coverage':pair['coverage'],'metrics':actual})
    require([p['sides'] for p in metrics['native_final_KV_pairs']]==KV_PAIRS,'four archived KV pair identities differ')
    for pair in metrics['native_final_KV_pairs']:
        require(pair['tensor_count']==len(pair['per_tensor'])==96 and pair['global']['count']==27525120 and
            set(pair['per_tensor'])=={'layer-%03d.%s'%(i,kind) for i in range(48) for kind in ('key','value')} and
            pair['thresholds'] is None and pair['acceptance'] is None,'archived complete96tensor KV scope differs')
    return {'status':'recorded_i03_target_metrics_replayed','target_pairs_checked':23,'pairs':checked,'independent_inputs':1,
        'target_metric_verification_tolerance':ATOL,'verification_tolerance_is_model_acceptance_threshold':False,
        'native_executions_started':0,'arrays_read':0,'archived_distribution_metrics_replayed':False,'archived_native_KV_metrics_replayed':False,
        'thresholds':None,'acceptance':None,'full_core_complete':False}


if __name__=='__main__':
    if len(sys.argv) not in (2,3):raise SystemExit('usage: python -S -B replay-targets.py selected-targets.json [selected-metrics.json]')
    try:result=replay(sys.argv[1],sys.argv[2] if len(sys.argv)==3 else None)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        print(json.dumps({'status':'rejected','reason':str(exc),'acceptance':None}));raise SystemExit(2)
    print(json.dumps(result,indent=2,allow_nan=False))
