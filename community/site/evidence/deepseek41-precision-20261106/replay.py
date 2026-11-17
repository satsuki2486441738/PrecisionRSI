from pathlib import Path
import argparse,json,hashlib,statistics,math
parser=argparse.ArgumentParser(description='Check published aggregate consistency; not raw numerical or model replay')
parser.add_argument('--directory',type=Path,default=Path(__file__).parent)
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args();root=a.directory;files={}
def read(name):
 p=root/name;files[name]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
d=read('core-measurements.json');g=read('gpu-performance.json');h=read('head-summary.json');gap=read('gap-ledger.json');runtime=read('runtime-facts.json')
assert d['schema_version']==1 and d['numerical_acceptance'] is False
assert d['all_agreed_numerical_chains_complete'] is False
stages=d['coverage'];assert len({x['stage'] for x in stages})==len(stages)==6
assert sum(x['expected'] for x in stages)==275
assert sum(x['paired'] for x in stages)==191
assert sum(x['gpu_missing'] for x in stages)==84
assert sum(x['npu_missing'] for x in stages)==6
assert d['targets']['target_count']==sum(x['target_count'] for x in d['target_groups'].values())==3168
assert len(d['candidate_rankings'])==6 and all(x['first_id_agreement'] for x in d['candidate_rankings'])
assert len(d['mhc_layers'])==40 and [x['layer'] for x in d['mhc_layers']]==list(range(40))
assert len(g['cells'])==9 and g['warmups_per_cell']==2 and g['measurements_per_cell']==5
seen=set()
for cell in g['cells']:
 key=(cell['kind'],cell['prompt_length']);assert key not in seen;seen.add(key)
 v=cell['measurements'];assert len(v)==5 and all(math.isfinite(x) and x>0 for x in v)
 assert statistics.median(v)==cell['median'] and min(v)==cell['minimum'] and max(v)==cell['maximum']
 assert math.isclose(statistics.mean(v),cell['arithmetic_mean'],rel_tol=1e-14)
 assert math.isclose(statistics.stdev(v),cell['stdev'],rel_tol=1e-12)
assert h['controls']==6 and h['changed_last_row_logits']==28 and h['max_logit_abs']==0.03125
assert h['generalization_to_core_None'] is False
assert len(gap)==11 and runtime['gpu']['torch']['value']=='2.13.0+cu130'

n=read('npu-performance.json');f=read('framework-facts.json');cleanup=read('terminal-cleanup-summary.json')
assert n['numerical_coverage'] is False and n['numerical_acceptance'] is False
expected={('prefill',x)for x in [128,512,2048,8192,32768]}|{('fixed32-cached-decode',x)for x in [128,512,2048,8192]}
assert {(c['kind'],c['prompt_length'])for c in n['cells']}==expected
assert len(n['cells'])==n['cells_count']==9
assert n['warmups_per_cell']==2 and n['measurements_per_cell']==5
prefills=cached=measured=warmups=0
for cell in n['cells']:
 assert len(cell['warmup_trials'])==2 and [x['trial']for x in cell['warmup_trials']]==[0,1]
 for container in [cell,cell['driver_generate_wall']]:
  v=container['measurements'];assert len(v)==5 and all(math.isfinite(x)and x>0 for x in v)
  assert statistics.median(v)==container['median'] and min(v)==container['minimum'] and max(v)==container['maximum']
  assert math.isclose(statistics.mean(v),container['arithmetic_mean'],rel_tol=1e-14)
  assert math.isclose(statistics.stdev(v),container['stdev'],rel_tol=1e-12)
 assert len(cell['request_prefill_measurements_seconds'])==5 and len(cell['sum_step_rank_max_diagnostic_seconds'])==5
 assert all(math.isfinite(x)and x>0 for x in cell['request_prefill_measurements_seconds']+cell['sum_step_rank_max_diagnostic_seconds'])
 for phase in ['phase','request']:
  assert 0<cell['max_measured_'+phase+'_peak_allocated_bytes']<=cell['max_measured_'+phase+'_peak_reserved_bytes']
 measured+=len(cell['measurements']);warmups+=len(cell['warmup_trials']);prefills+=7
 if cell['kind']=='fixed32-cached-decode':cached+=7*32
assert measured==n['measured_requests']==45 and warmups==n['warmup_requests']==18
assert prefills==n['requests']==n['prefills_per_rank']==63
assert cached==n['cached_forwards_per_rank']==896
assert prefills+cached==n['forwards_per_rank']==959
assert prefills==n['untimed_cleanup_ticks_per_rank']==63
counts=cleanup['receipt_counts']
assert set(counts['per_rank_forward_counts'])==set(counts['per_rank_empty_counts'])==set(map(str,range(8)))
assert set(counts['per_rank_forward_counts'].values())=={959} and set(counts['per_rank_empty_counts'].values())=={63}
assert counts['verified_files']==cleanup['transport_files_rehashed']==433 and counts['verified_bytes']==cleanup['transport_bytes_rehashed']==432763480
assert cleanup['terminal_returncode']==0 and cleanup['owner_cleanup']['recorded_terminal_group_absent']
assert cleanup['recorded_device_cleanup']['aicore_zero'] and cleanup['recorded_device_cleanup']['no_processes_on_8boards']
assert len(f)==8 and {(x['side'],x['family'])for x in f}=={(side,family)for side in ['gpu','npu']for family in ['trainer','sfttrainer','grpo','verl']}
assert all(x['accelerator_execution'] is False and x['numerical_coverage'] is False for x in f)
manifest=json.loads((root/'manifest.json').read_text())
assert set(manifest['files'])=={p.name for p in root.iterdir()if p.is_file()and p.name!='manifest.json'}
for name,entry in manifest['files'].items():
 p=root/name;assert p.stat().st_size==entry['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']

r=dict(status='passed-aggregate-consistency-only',raw_numerical_replay=False,model_executed=False,precision_acceptance=False,paired_tensor_cells=191,expected_tensor_cells=275,target_coordinates=3168,gpu_measured_trials=45,npu_measured_trials=measured,npu_requests=prefills,npu_forwards_per_rank=prefills+cached,npu_cleanup_ticks_per_rank=prefills,manifest_verified=True,files_sha256=files)
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
