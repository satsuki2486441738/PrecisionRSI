"""Replay six anonymous GPU observer target pairs using stdlib only."""
import hashlib,json,math,pathlib,sys
def require(ok,message):
 if not ok:raise ValueError(message)
def read(path):
 p=pathlib.Path(path);require(p.suffix=='.json' and p.stat().st_size<4*1024**2,'bounded JSON only')
 raw=p.read_bytes();return raw,json.loads(raw,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def reduce(a,b):
 require(len(a)==len(b)==159 and all(type(x) in (int,float) and math.isfinite(x) and x<=0 for x in a+b),'159 finite logprobs')
 e=[abs(x-y) for x,y in zip(a,b)];s=sorted(e);h=.99*158;lo=int(h);na,nb=[math.fsum(-x for x in v)/159 for v in (a,b)]
 return {'target_count':159,'side_a_nll_nat_per_token':na,'side_b_nll_nat_per_token':nb,'nll_abs_diff_nat_per_token':abs(na-nb),'mae_nat':math.fsum(e)/159,'p99_nat':s[lo]+(h-lo)*(s[lo+1]-s[lo]),'max_abs_nat':max(e),'max_target_index':e.index(max(e))}
raw,targets=read(sys.argv[1]);_,metrics=read(sys.argv[2] if len(sys.argv)>2 else pathlib.Path(sys.argv[1]).with_name('selected-metrics.json'))
require(hashlib.sha256(raw).hexdigest()==metrics['selected_targets_sha256'],'target byte SHA differs')
require(targets['record_id']==metrics['record_id']=='gemma4-h100-observer-transparency-20261107' and targets['acceptance'] is metrics['acceptance'] is None and metrics['full_core_complete'] is False,'record/scope differs')
vectors={row['id']:row['logprobs'] for row in targets['vectors']};require(len(vectors)==4 and len(metrics['target_distribution_pairs'])==6,'four vectors/six pairs required')
for pair in metrics['target_distribution_pairs']:
 a,b=pair['sides'];actual=reduce(vectors[a],vectors[b])
 for key,value in actual.items():require(abs(value-pair['targets'][key])<=1e-12,'target reduction mismatch '+key)
print(json.dumps({'status':'six_target_pairs_replayed','pairs':6,'arrays_opened':0,'verification_atol':1e-12,'acceptance':None,'distribution_or_tensor_values_replayed':False},indent=2))
