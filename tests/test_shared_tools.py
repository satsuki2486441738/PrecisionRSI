"""Regressions for framework-independent score and condition tools."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]/'skills/precision-skill/scripts'
sys.path.insert(0,str(SCRIPTS))
from score_metrics import reduce_scores
from compare_target_scores import compare as target_scores


class ScoreMetricsTests(unittest.TestCase):
    def test_target_weighting_ties_and_order(self):
        a={('a',0,10):-1.,('b',0,11):-2.,('b',1,12):-3.}
        b={('b',1,12):-5.,('a',0,10):-3.,('b',0,11):-2.}
        r=reduce_scores(a,b)
        self.assertAlmostEqual(r['side_a_nll_nat_per_token'],2.)
        self.assertAlmostEqual(r['side_b_nll_nat_per_token'],10/3)
        self.assertAlmostEqual(r['mae_nat'],4/3)
        self.assertEqual(r['max_coordinate'],['a',0,10])
        self.assertEqual(r['max_abs_nat'],2.)
        swapped=reduce_scores(b,a)
        for key in ('mae_nat','p99_nat','max_coordinate','max_abs_nat','nll_abs_diff_nat_per_token'):
            self.assertEqual(r[key],swapped[key])
        self.assertEqual(r,reduce_scores(dict(reversed(list(a.items()))),b))

    def test_score_validation_and_single_target(self):
        valid={('x',0,1):-1.}
        for a,b in [({},{}),(valid,{}),(valid,{('x',0,2):-1.}),
                    (valid,{('x',0,1):float('nan')}),(valid,{('x',0,1):True}),
                    (valid,{('x',0,1):0.1})]:
            with self.subTest(a=a,b=b),self.assertRaises(ValueError):reduce_scores(a,b)
        r=reduce_scores(valid,{('x',0,1):-3.})
        self.assertEqual((r['p99_nat'],r['mae_nat'],r['max_abs_nat']),(2.,2.,2.))


class TargetArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.common=dict(model='m',inputs='i',scope='training-active-eos')
        self.spec=dict(schema_version=1,kind='target-scores',thresholds=None,common=self.common,
            side_conditions={'a':{'numerics':'float16'},'b':{'numerics':'bfloat16'}},
            expected_ranks={'a':[0],'b':[0]},targets=[dict(sample='x',position=7,token=2)])
        self.write('contract.json',self.spec)
        self.manifests={}
        for side,value in [('a',-1.),('b',-3.)]:
            self.write(side+'-env.json',dict(rank=0,versions={'framework':'test'},
                actual={**self.common,**self.spec['side_conditions'][side]}))
            self.write(side+'-status.json',dict(rank=0,status='completed'))
            m=dict(schema_version=1,kind='target-scores',common=self.common,side=side,expected_ranks=[0],
                ranks=[dict(rank=0,environment=side+'-env.json',status=side+'-status.json')],
                scores=[dict(sample='x',position=7,token=2,logprob=value)])
            self.manifests[side]=m;self.write(side+'.json',m)

    def write(self,name,value):
        (self.root/name).write_text(json.dumps(value))

    def compare(self):
        return target_scores(self.root/'contract.json',self.root/'a.json',self.root/'b.json')

    def test_training_eos_and_per_side_precision(self):
        r=self.compare()
        self.assertEqual(r['status'],'no_thresholds')
        self.assertEqual(r['metrics']['max_abs_nat'],2.)
        self.assertEqual(r['metrics']['max_coordinate'],dict(sample='x',position=7,token=2))
        self.assertEqual(r['coverage'],dict(expected=1,paired=1))

    def test_actual_setting_and_rank_coverage(self):
        self.write('b-env.json',dict(rank=0,versions={'f':'v'},actual={**self.common,'numerics':'wrong'}))
        self.assertEqual(self.compare()['status'],'incomparable')
        self.write('b-env.json',dict(rank=0,versions={'f':'v'},actual={**self.common,'numerics':'bfloat16'}))
        self.manifests['b']['expected_ranks']=[];self.write('b.json',self.manifests['b'])
        self.assertEqual(self.compare()['status'],'not_covered')

    def test_nonobject_json_returns_diagnostics(self):
        self.write('b.json',[])
        self.assertEqual(self.compare()['status'],'incomparable')
        self.write('contract.json',[])
        self.assertEqual(self.compare()['status'],'incomparable')

    def test_missing_or_duplicate_coordinates(self):
        m=self.manifests['b']
        m['scores'].append(copy.deepcopy(m['scores'][0]));self.write('b.json',m)
        self.assertEqual(self.compare()['status'],'incomparable')
        m['scores']=[];self.write('b.json',m)
        self.assertEqual(self.compare()['status'],'not_covered')

    def test_cli_output_and_exclusive_creation(self):
        out=self.root/'result.json'
        command=[sys.executable,'-B','-I',str(SCRIPTS/'compare_target_scores.py'),
                 '--contract',str(self.root/'contract.json'),'--side-a',str(self.root/'a.json'),
                 '--side-b',str(self.root/'b.json'),'--output',str(out)]
        result=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        before=out.read_bytes()
        self.assertEqual(subprocess.run(command,capture_output=True).returncode,2)
        self.assertEqual(out.read_bytes(),before)


class DistributionQueryTests(unittest.TestCase):
    def fixture(self):
        from test_artifact_tools import Fixture
        c=Fixture();c.setUp();self.addCleanup(c.doCleanups)
        spec,a,b=c.dist_pair()
        cp=c.write('contract.json',spec)
        ap=c.write('a.json',a);bp=c.write('b.json',b)
        return c,cp,ap,bp

    def test_union_queries_equal_full_array_metrics(self):
        from query_distributions import inventory,query,compare
        from compare_distributions import compare as full
        c,cp,ap,bp=self.fixture()
        ia,ib=inventory(cp,ap),inventory(cp,bp)
        qa,qb=query(cp,ap,ib),query(cp,bp,ia)
        got=compare(cp,qa,qb);expected=full(cp,ap,bp)
        self.assertEqual(got['aggregate'],expected['aggregate'])
        self.assertEqual(got['position_count'],expected['position_count'])
        altered=copy.deepcopy(qb)
        altered['positions'][0]['tail_logmass']=-0.5
        with self.assertRaises(ValueError):compare(cp,qa,altered)
        swapped=compare(cp,qb,qa)
        for field in ('tail_merged_js','probability_rmse','max_probability_error','top1_agreement'):
            self.assertEqual(got['aggregate'][field],swapped['aggregate'][field])
        broken=copy.deepcopy(qb);broken['peer_source_id']='wrong'
        with self.assertRaises(ValueError):compare(cp,qa,broken)
        broken=copy.deepcopy(qb);broken['positions'][0]['union'].pop()
        with self.assertRaises(ValueError):compare(cp,qa,broken)

    def test_query_payload_substitution_is_rejected(self):
        import numpy as np
        from test_artifact_tools import Fixture
        from query_distributions import inventory,query,compare
        c=Fixture();c.setUp();self.addCleanup(c.doCleanups)
        p=np.arange(202,0,-1,dtype=np.float64);p/=p.sum()
        q=p.copy();q[0]-=1e-6;q[1]+=1e-6
        spec,a,b=c.dist_pair(p,q)
        cp=c.write('contract.json',spec);ap=c.write('a.json',a);bp=c.write('b.json',b)
        ia,ib=inventory(cp,ap),inventory(cp,bp)
        qa,qb=query(cp,ap,ib),query(cp,bp,ia)
        self.assertGreater(compare(cp,qa,qb)['aggregate']['max_probability_error']['max'],0.)
        mixed=copy.deepcopy(qb);mixed['positions']=copy.deepcopy(qa['positions'])
        with self.assertRaises(ValueError):compare(cp,qa,mixed)

    def test_zero_probabilities_and_changed_source_binding(self):
        import numpy as np
        from test_artifact_tools import Fixture
        from query_distributions import inventory,query,compare
        c=Fixture();c.setUp();self.addCleanup(c.doCleanups)
        values=np.zeros(101);values[0]=1
        spec,a,b=c.dist_pair(values,values)
        cp=c.write('contract.json',spec);ap=c.write('a.json',a);bp=c.write('b.json',b)
        ia,ib=inventory(cp,ap),inventory(cp,bp)
        qa,qb=query(cp,ap,ib),query(cp,bp,ia)
        json.dumps(qa,allow_nan=False)
        self.assertEqual(compare(cp,qa,qb)['aggregate']['tail_merged_js']['mean'],0.)
        path=c.root/b['positions'][0]['scores']['path']
        altered=np.full((1,101),-np.inf);altered[0,1]=0.
        np.save(path,altered)
        new=query(cp,bp,ia)
        with self.assertRaises(ValueError):compare(cp,qa,new)


class TensorArtifactTests(unittest.TestCase):
    def test_arbitrary_rank_shards_and_explicit_replica_views(self):
        import numpy as np
        from test_artifact_tools import Fixture
        from tensor_artifacts import export_manifest,compare_views
        from compare_tensors import compare
        c=Fixture();c.setUp();self.addCleanup(c.doCleanups)
        spec,_,_=c.tensor_pair(r=[1.,2.,3.],c=[1.,2.,3.])
        spec['expected_ranks']={'a':[0,1,2],'b':[0,1,2]}
        cp=c.write('contract.json',spec)
        paths=[]
        for side in ('a','b'):
            ranks=[]
            for rank in range(3):
                c.write(f'{side}-env{rank}.json',dict(rank=rank,versions={'framework':'test'}))
                c.write(f'{side}-status{rank}.json',dict(rank=rank,status='completed'))
                c.write(f'{side}-inventory{rank}.json',[dict(name='weight',shape=[3],present=True,
                    shards=[dict(offset=rank,value=c.array([rank+1.],np.float16))])])
                ranks.append(dict(rank=rank,environment=f'{side}-env{rank}.json',status=f'{side}-status{rank}.json',inventory=f'{side}-inventory{rank}.json'))
            capture=c.write(side+'-capture.json',dict(side=side,common=spec['common'],expected_ranks=[0,1,2],stage='gradients',ranks=ranks))
            m=export_manifest(cp,capture)
            paths.append(c.write(side+'-logical.json',m))
        self.assertEqual(compare(cp,*paths)['global']['max_abs']['value'],0.)
        result=compare_views(cp,[dict(view='logical',side_a=str(paths[0]),side_b=str(paths[1]))])
        self.assertEqual(result['status'],'no_thresholds')
        self.assertNotIn('global',result)
        with self.assertRaises(ValueError):compare_views(cp,[dict(view='wrong',side_a=str(paths[0]),side_b=str(paths[1]))])
        with self.assertRaises(ValueError):compare_views(cp,[
            dict(view='first',side_a=str(paths[0]),side_b=str(paths[1]),side_a_view='logical',side_b_view='logical'),
            dict(view='second',side_a=str(paths[0]),side_b=str(paths[1]),side_a_view='logical',side_b_view='logical')])
        # Two complete replicas, with an explicit view for their differing values.
        spec['expected_ranks']={'a':[0,1]};spec['tensors'][0]['shape']=[2];c.write('contract.json',spec)
        ranks=[]
        for rank in (0,1):
            shards=[dict(offset=0,value=c.array([1.,2.+rank]),view=f'r{rank}')]
            c.write(f'replica{rank}.json',[dict(name='weight',shape=[2],present=True,shards=shards)])
            ranks.append(dict(rank=rank,environment=f'a-env{rank}.json',status=f'a-status{rank}.json',inventory=f'replica{rank}.json'))
        capture=c.write('replicas.json',dict(side='a',common=spec['common'],expected_ranks=[0,1],stage='gradients',ranks=ranks))
        with self.assertRaises(ValueError):export_manifest(cp,capture)
        first=export_manifest(cp,capture,'r0');second=export_manifest(cp,capture,'r1')
        self.assertNotEqual(first['tensors'][0]['shards'],second['tensors'][0]['shards'])
        self.assertEqual(len(first['ranks']),2)
        c.write('replica1.json',[dict(name='weight',shape=[2],present=True,
            shards=[dict(offset=0,value=c.array([1.,2.]),view='r1')])])
        equal=export_manifest(cp,capture)
        self.assertEqual(len(equal['tensors'][0]['shards']),1)
