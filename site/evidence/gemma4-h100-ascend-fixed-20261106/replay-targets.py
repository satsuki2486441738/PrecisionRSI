#!/usr/bin/env python3
"""Replay all 15 anonymous recorded target comparisons using the stdlib.

Usage: python replay-targets.py selected-targets.json [selected-metrics.json]
The second JSON defaults to the sibling selected-metrics.json. Both immutable
public artifacts are hash checked. Distribution summaries are integrity checked,
not numerically replayed. No full-vocabulary array is opened by this program.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

TARGETS_SHA256='d3436c2eff533ccd2b866b10839ab8ab21170f8456312c925a9b8dae6d1b5876'
METRICS_SHA256='6f63c2356fa8951e2181aaecdc9136826c76befb2d141cb95da40384c74bf4cc'
SIDES=['h100-r1','h100-r2','h100-r3','npu-r1','npu-r2','npu-r3']
ATOL=1e-12


def require(ok,message):
    if not ok:raise ValueError(message)


def load_frozen(path,digest,label):
    path=Path(path)
    require(path.is_file() and path.stat().st_size<=1024**2,'bounded JSON artifact required: '+label)
    raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==digest,'frozen '+label+' bytes differ')
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate JSON key');result[key]=value
        return result
    def reject(value):raise ValueError('nonfinite JSON constant: '+value)
    doc=json.loads(raw,object_pairs_hook=pairs,parse_constant=reject)
    json.dumps(doc,allow_nan=False)
    return doc


def metrics(a,b):
    errors=[abs(x-y) for x,y in zip(a,b)]
    ordered=sorted(errors);offset=.99*(len(errors)-1);low=math.floor(offset);high=math.ceil(offset)
    na,nb=(math.fsum(-value for value in side)/len(side) for side in (a,b))
    return {'target_count':len(errors),'side_a_nll_nat_per_token':na,'side_b_nll_nat_per_token':nb,
        'nll_abs_diff_nat_per_token':abs(na-nb),'mae_nat':math.fsum(errors)/len(errors),
        'p99_nat':ordered[low]+(offset-low)*(ordered[high]-ordered[low]),
        'p99_method':'global linear h=0.99*(N-1)','max_abs_nat':max(errors),
        'max_anonymous_target_index_zero_based':errors.index(max(errors))}


def replay(target_path,metric_path=None):
    target_path=Path(target_path)
    metric_path=Path(metric_path) if metric_path else target_path.with_name('selected-metrics.json')
    data=load_frozen(target_path,TARGETS_SHA256,'selected-targets')
    archived=load_frozen(metric_path,METRICS_SHA256,'selected-metrics')
    require(data.get('schema_version')==archived.get('schema_version')==1 and
        data.get('kind')=='anonymous_gpu_npu_fixed_target_replay' and
        archived.get('kind')=='anonymous_gpu_npu_fixed_comparison_metrics','unsupported public evidence schema')
    for doc in (data,archived):
        require(doc.get('status')=='scoped_measured_fixed_input_comparison' and doc.get('thresholds') is None and
            doc.get('acceptance') is None and doc.get('full_core_complete') is False,'unexpected acceptance/completion claim')
        require(doc['model']['repository']=='google/gemma-4-12B' and doc['model']['variant']=='base' and
            doc['model']['revision']=='1a2c7729392586f5ecddae8807ee2ef741eabb9a' and
            doc['input_sha256']=='643d29f23d00f97665c5c72d73c63bc20525950b2ca5c50b435140b6066214f0',
            'fixed model/input identity differs')
    for key in ('model','input_sha256','workload','array_evidence','source_pins','target_pairs'):
        require(data[key]==archived[key],'public evidence differs: '+key)
    expected_workload={'parameter_dtype':'BF16','attention':'eager','seed':0,'parallelism':'TP1',
        'independent_inputs':1,'conditioning_tokens_including_bos':128,'fixed_continuation_tokens':32,
        'fresh_process_repeats_per_backend':3,'total_captures':6,'prefill_calls_per_capture':1,
        'cached_single_token_decode_calls_per_capture':31,'target_positions_per_capture':32,'vocabulary_size':262144,
        'native_logits_retained':False}
    require(all(k in data['workload'] and type(data['workload'][k]) is type(v) and data['workload'][k]==v
        for k,v in expected_workload.items()),'workload scope differs')
    require(data['array_evidence']=={
        'source_gate_full_vocabulary_rows_checked':{'GPU':96,'NPU':96},
        'full_vocabulary_rows_available_on_delivery_machine':{'GPU':96,'NPU':0},
        'metadata_export_full_vocabulary_rows_transferred':0,'public_target_replay_full_vocabulary_rows_read':0,
        'public_target_replay_recomputes_distribution_metrics':False},'array availability versus replay scope differs')
    require(archived['target_replay_artifact']=={'file':'selected-targets.json','sha256':TARGETS_SHA256},
        'archived target artifact reference differs')
    require([r['id'] for r in data['runs']]==SIDES,'six ordered anonymous captures required')
    vectors={}
    for run in data['runs']:
        values=run['target_logprobs_nat']
        require(run['backend']==('GPU' if run['id'].startswith('h100-') else 'NPU') and
            isinstance(values,list) and len(values)==32 and
            all(type(v) in (int,float) and math.isfinite(v) and v<=0 for v in values),'32 finite recorded target values required')
        require(len(run['ordered_raw_record_sha256'])==32,'original record digest coverage differs')
        vectors[run['id']]=values
    expected_pairs=[list(pair) for pair in itertools.combinations(SIDES,2)]
    require([p['sides'] for p in data['target_pairs']]==expected_pairs and
        [p['sides'] for p in archived['distribution_pairs']]==expected_pairs,'all 15 ordered capture pairs required')
    require(archived['pair_scope']=={'cross_backend_pairs':9,'within_gpu_pairs':3,'within_npu_pairs':3,
        'all_pairs':15,'cross_pairs_share_one_input':True},'pair scope differs')
    independent=archived['independent_verification']
    require(independent['target_pairs_checked']==15 and independent['distribution_pair_positions_checked']==32 and
        independent['prespecified_distribution_pair']==['h100-r1','npu-r1'] and
        independent['anonymous_prediction_indices_zero_based']==list(range(32)) and
        independent['all_192_distribution_rows_independently_replayed'] is False and independent['native_executions_started']==0,
        'independent selected-row versus full-gate scope differs')
    checked=[]
    for pair in data['target_pairs']:
        a,b=pair['sides'];kind='within_backend_repeat' if a.split('-')[0]==b.split('-')[0] else 'cross_backend'
        require(pair['kind']==kind and pair['coverage']=={'expected':32,'paired':32},'target pair kind/coverage differs')
        actual=metrics(vectors[a],vectors[b]);recorded=pair['metrics']
        require(set(actual)==set(recorded),'recorded target metric inventory differs')
        for key,value in actual.items():
            reference=recorded[key]
            if type(value) is str:require(reference==value,'quantile method differs')
            elif type(value) is int:require(type(reference) is int and reference==value,'integer target metric differs: '+key)
            else:require(type(reference) in (int,float) and math.isfinite(reference) and
                math.isclose(value,reference,rel_tol=0,abs_tol=ATOL),'target metric differs: '+key)
        checked.append({'sides':[a,b],'kind':kind,'coverage':pair['coverage'],'metrics':actual})
    return {'status':'recorded_target_metrics_replayed','selected_targets_sha256':TARGETS_SHA256,
        'selected_metrics_sha256':METRICS_SHA256,'pairs':checked,'independent_inputs':1,
        'absolute_metric_verification_tolerance':ATOL,'verification_tolerance_is_model_acceptance_threshold':False,
        'full_vocabulary_rows_read':0,'archived_distribution_metrics_replayed':False,
        'full_vocabulary_rows_available_on_delivery_machine':{'GPU':96,'NPU':0},
        'thresholds':None,'acceptance':None,'full_core_complete':False}


if __name__=='__main__':
    if len(sys.argv) not in (2,3):raise SystemExit('usage: python replay-targets.py selected-targets.json [selected-metrics.json]')
    try:result=replay(sys.argv[1],sys.argv[2] if len(sys.argv)==3 else None)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        print(json.dumps({'status':'rejected','reason':str(exc),'acceptance':None,'full_core_complete':False}))
        raise SystemExit(2)
    print(json.dumps(result,indent=2,allow_nan=False))
