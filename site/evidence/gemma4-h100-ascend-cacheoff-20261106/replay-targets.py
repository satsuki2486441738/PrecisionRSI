#!/usr/bin/env python3
"""Replay all7 anonymous I01 target comparisons with the Python stdlib.

Usage: python -S -B replay-targets.py selected-targets.json [selected-metrics.json]
Frozen distribution summaries are integrity checked, not numerically replayed.
This program opens no raw native tensor or full-vocabulary array.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

TARGETS_SHA256='7e52fb4b812924ea08a9ff10e9708a16f4a9503f4c965a35bd0f01658361de63'
METRICS_SHA256='b9646885fdebc1693a31738dd41389265a13e8145e18f8cd61c8ff5ec9ef2970'
SIDES=['gpu-full','npu-full','gpu-cached-r1','gpu-cached-r2','gpu-cached-r3','npu-cached-r1','npu-cached-r2','npu-cached-r3']
PAIRS=[['gpu-full','npu-full']]+[['gpu-full','gpu-cached-r%d'%i] for i in (1,2,3)]+[['npu-full','npu-cached-r%d'%i] for i in (1,2,3)]
ATOL=1e-12


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def load_frozen(path,digest):
    path=Path(path);require(path.is_file() and path.stat().st_size<=2*1024**2,'bounded frozen JSON required')
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==digest,'frozen public artifact SHA differs')
    def pairs(items):
        out={}
        for key,value in items:require(key not in out,'duplicate JSON key');out[key]=value
        return out
    def reject(value):raise ValueError('nonfinite JSON token '+value)
    doc=json.loads(raw,object_pairs_hook=pairs,parse_constant=reject);canonical(doc);return doc


def calculate(a,b,start):
    require(len(a)==len(b) and len(a) in (32,159),'paired target count differs')
    errors=[abs(x-y) for x,y in zip(a,b)];ordered=sorted(errors);offset=.99*(len(errors)-1)
    lo,hi=math.floor(offset),math.ceil(offset);na,nb=[math.fsum(-v for v in values)/len(values) for values in (a,b)]
    maximum=max(errors);index=errors.index(maximum)
    return {'target_count':len(errors),'side_a_nll_nat_per_token':na,'side_b_nll_nat_per_token':nb,
        'nll_abs_diff_nat_per_token':abs(na-nb),'mae_nat':math.fsum(errors)/len(errors),
        'p99_nat':ordered[lo]+(offset-lo)*(ordered[hi]-ordered[lo]),'p99_method':'global linear h=0.99*(N-1)',
        'max_abs_nat':maximum,'max_target_index_zero_based':start+index,'max_target_index_within_pair_zero_based':index}


def replay(target_path,metric_path=None):
    target_path=Path(target_path);metric_path=Path(metric_path) if metric_path else target_path.with_name('selected-metrics.json')
    targets=load_frozen(target_path,TARGETS_SHA256);metrics=load_frozen(metric_path,METRICS_SHA256)
    require(targets['kind']=='anonymous_i01_target_replay' and metrics['kind']=='anonymous_i01_comparison_metrics','unsupported evidence kind')
    for doc in (targets,metrics):
        require(doc['schema_version']==1 and doc['status']=='scoped_measured_fixed_input_comparison' and doc['thresholds'] is None and
            doc['acceptance'] is None and doc['full_core_complete'] is False,'unexpected completion/acceptance claim')
        require(doc['model']=={'repository':'google/gemma-4-12B','variant':'base','revision':'1a2c7729392586f5ecddae8807ee2ef741eabb9a',
            'parameter_dtype':'BF16','attention':'eager'},'fixed model differs')
        workload=doc['workload'];require(workload['independent_inputs']==1 and workload['new_captures_per_backend']==1 and
            workload['new_repeatability_measured'] is False and workload['new_use_cache'] is False and workload['targets_per_new_capture']==159 and
            workload['distribution_rows_per_new_capture']==47 and workload['native_logits_shape_per_new_capture']==[1,160,262144] and
            workload['unscored_prediction_row159_retained_and_finite'] is True,'measured workload scope differs')
        require(doc['pair_scope']=={'cross_backend_full_pairs':1,'historical_gpu_peer_pairs':3,'historical_npu_peer_pairs':3,'total_pairs':7,
            'all_pairs_share_one_input':True},'seven shared-input comparisons required')
        require(doc['array_evidence']['public_target_replay_arrays_read']==0 and doc['array_evidence']['public_target_replay_recomputes_distribution_metrics'] is False,
            'target-only replay scope differs')
    for key in ('model','source_pins','workload','pair_scope','coordinate_convention','array_evidence','target_pairs'):
        require(targets[key]==metrics[key],'public artifacts disagree: '+key)
    require(metrics['target_replay_artifact']=={'file':'selected-targets.json','sha256':TARGETS_SHA256},'target artifact binding differs')
    require([run['id'] for run in targets['runs']]==SIDES and [run['id'] for run in metrics['runs']]==SIDES,'eight ordered anonymous vectors required')
    vectors={}
    for ordinal,run in enumerate(targets['runs']):
        start,count=(0,159) if ordinal<2 else (127,32)
        values=run['target_logprobs_nat'];require(run['target_start_index_zero_based']==start and run['target_count']==count and
            isinstance(values,list) and len(values)==count and all(type(v) in (int,float) and math.isfinite(v) and v<=0 for v in values),'target vector domain/values differ')
        require(hashlib.sha256(canonical(values).encode()).hexdigest()==run['target_vector_sha256'],'anonymous vector SHA differs')
        if ordinal>=2:require(run['historical_published_vector_preserved_exactly'] is True,'historical32 values must be preserved')
        require({k:v for k,v in run.items() if k!='target_logprobs_nat'}==metrics['runs'][ordinal],'run metadata differs')
        vectors[run['id']]={index:value for index,value in enumerate(values,start)}
    require([p['sides'] for p in targets['target_pairs']]==PAIRS and [p['sides'] for p in metrics['distribution_pairs']]==PAIRS,'ordered seven pair identities differ')
    checked=[]
    for index,pair in enumerate(targets['target_pairs']):
        a,b=pair['sides'];start,count=(0,159) if index==0 else (127,32)
        require(pair['target_start_index_zero_based']==start and pair['coverage']=={'expected':count,'paired':count} and
            pair['kind']==('cross_backend_full' if index==0 else 'historical_cached_peer'),'159/full versus32/continuation pairing differs')
        if index:
            peer=pair['peer_context'];require(peer['controlled_cache_only_intervention'] is False and
                peer['same_physical_device_identifier'] is (index>=4),'historical device/cache confound differs')
        actual=calculate([vectors[a][i] for i in range(start,start+count)],[vectors[b][i] for i in range(start,start+count)],start)
        require(set(actual)==set(pair['metrics']),'target metric field inventory differs')
        for key,value in actual.items():
            reference=pair['metrics'][key]
            if type(value) in (int,str):require(type(reference) is type(value) and reference==value,'discrete target metric differs: '+key)
            else:require(type(reference) in (int,float) and math.isfinite(reference) and math.isclose(value,reference,rel_tol=0,abs_tol=ATOL),'target scalar differs: '+key)
        distribution=metrics['distribution_pairs'][index]
        positions=[0,8,16,25,33,42,50,59,67,76,84,93,101,110,118]+list(range(127,159)) if index==0 else list(range(127,159))
        require(distribution['position_count']==len(positions) and [row['prediction_index_zero_based'] for row in distribution['positions']]==positions and
            [row['sampled_position_index_zero_based'] for row in distribution['positions']]==list(range(len(positions))),'archived distribution coordinate inventory differs')
        require(distribution['aggregate']['top1_agreement']['total']==len(positions),'archived top1 coverage differs')
        checked.append({'sides':[a,b],'coverage':pair['coverage'],'metrics':actual})
    return {'status':'recorded_i01_target_metrics_replayed','target_pairs_checked':7,'pairs':checked,'independent_inputs':1,
        'target_metric_verification_tolerance':ATOL,'verification_tolerance_is_model_acceptance_threshold':False,
        'native_executions_started':0,'arrays_read':0,'archived_distribution_metrics_replayed':False,
        'thresholds':None,'acceptance':None,'full_core_complete':False}


if __name__=='__main__':
    if len(sys.argv) not in (2,3):raise SystemExit('usage: python -S -B replay-targets.py selected-targets.json [selected-metrics.json]')
    try:result=replay(sys.argv[1],sys.argv[2] if len(sys.argv)==3 else None)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        print(json.dumps({'status':'rejected','reason':str(exc),'acceptance':None}));raise SystemExit(2)
    print(json.dumps(result,indent=2,allow_nan=False))
