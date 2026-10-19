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


class DistributionRowTests(unittest.TestCase):
    def fixture(self):
        import numpy as np
        from test_artifact_tools import Fixture
        c=Fixture();c.setUp();self.addCleanup(c.doCleanups)
        p=np.full((2,101),.5/100);p[0,0]=.5;p[1,1]=.5;p[1,0]=.5/100
        q=p.copy();q[1,[1,2]]=q[1,[2,1]]
        spec,a,b=c.dist_pair(p[0],q[0])
        spec['positions']=[dict(sample='x',position=i,history=[1]*(i+1)) for i in range(2)]
        for m,values in [(a,p),(b,q)]:
            descriptor=c.array(np.log(values))
            m['positions']=[dict(**coord,scores={**descriptor,'row':i}) for i,coord in enumerate(spec['positions'])]
        return c,spec,a,b

    def test_distinct_rows_in_full_validation_and_query_paths(self):
        from query_distributions import inventory,query,compare
        c,spec,a,b=self.fixture();r=c.run_dist(spec,a,b)
        self.assertEqual(r['status'],'no_thresholds')
        self.assertEqual(r['positions'][0]['max_probability_error'],0.)
        self.assertGreater(r['positions'][1]['max_probability_error'],.49)
        cp=c.root/'contract.json';ap=c.root/'gpu.json';bp=c.root/'npu.json'
        ia,ib=inventory(cp,ap),inventory(cp,bp)
        self.assertEqual([p['source']['row'] for p in ia['positions']],[0,1])
        self.assertEqual(compare(cp,query(cp,ap,ib),query(cp,bp,ia))['aggregate'],r['aggregate'])

    def test_misplaced_and_missing_multirow_indexes_are_rejected(self):
        from validate_artifacts import validate
        from query_distributions import inventory
        for misplaced in (True,False):
            with self.subTest(misplaced=misplaced):
                c,spec,a,b=self.fixture()
                for row in b['positions']:
                    i=row['scores'].pop('row')
                    if misplaced:row['row']=i
                self.assertEqual(c.run_dist(spec,a,b)['status'],'incomparable')
                cp=c.root/'contract.json';bp=c.root/'npu.json'
                self.assertEqual(validate(cp,bp)['status'],'incomparable')
                with self.assertRaisesRegex(ValueError,'scores'):inventory(cp,bp)

    def test_legacy_single_row_and_invalid_row_types(self):
        from compare_distributions import scores
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);np.save(p/'single.npy',np.log(np.full((1,100),.01)))
            scores(p,{'scores':{'path':'single.npy'}},100,1e-6)
            for row in (True,-1,1,'0'):
                with self.subTest(row=row),self.assertRaises(ValueError):
                    scores(p,{'scores':{'path':'single.npy','row':row}},100,1e-6)


class ObservationTests(unittest.TestCase):
    def fixture(self):
        from test_artifact_tools import Fixture
        c=Fixture();c.setUp();self.addCleanup(c.doCleanups)
        spec,a,b=c.tensor_pair(r=[1.,2.,3.,4.],c=[1.,2.,3.,6.],stage='observations')
        del spec['common']['state']
        coords=dict(sample='x',tokens=[0,1],boundary='block.3.output',invocation=0,step=0,direction='forward')
        for item in (spec,a,b):
            item['schema_version']=2
            item['common']=copy.deepcopy(spec['common'])
            item['tensors'][0].update(name='block3-output',shape=[1,2,2],coordinates=copy.deepcopy(coords))
        for side in ('GPU','NPU'):
            c.write(side+'-env.json',dict(rank=0,versions={'framework':'test'},actual=spec['common']))
        return c,spec,a,b

    def test_activation_metrics_and_multidimensional_coordinates(self):
        from validate_artifacts import validate
        c,spec,a,b=self.fixture();r=c.run_tensors(spec,a,b)
        self.assertEqual(r['status'],'no_thresholds');self.assertEqual(r['stage'],'observations')
        self.assertAlmostEqual(r['global']['rmse']['value'],1.)
        self.assertEqual(r['global']['max_coordinate']['tensor'],'block3-output')
        self.assertEqual(r['global']['max_coordinate']['coordinate'],[0,1,1])
        self.assertEqual(r['per_tensor']['block3-output']['coordinates'],spec['tensors'][0]['coordinates'])
        self.assertEqual(validate(c.root/'contract.json',c.root/'gpu.json')['coordinate_count'],4)
        json.dumps(r,allow_nan=False)

    def test_missing_or_different_observation_coordinates(self):
        c,spec,a,b=self.fixture();b['tensors'][0]['coordinates']['tokens']=[1,2]
        r=c.run_tensors(spec,a,b)
        self.assertEqual(r['status'],'incomparable');self.assertEqual(r['global']['status'],'not_covered')
        c,spec,a,b=self.fixture();del spec['tensors'][0]['coordinates']
        self.assertEqual(c.run_tensors(spec,a,b)['status'],'incomparable')
        c,spec,a,b=self.fixture();spec['tensors'][0]['present']=False
        self.assertEqual(c.run_tensors(spec,a,b)['status'],'incomparable')


    def test_observation_export_preserves_and_checks_coordinates(self):
        from tensor_artifacts import export_manifest
        c,spec,a,b=self.fixture();cp=c.write('contract.json',spec)
        inv=c.write('inventory.json',a['tensors'])
        capture={**a,'ranks':[{**a['ranks'][0],'inventory':inv.name}]}
        path=c.write('capture.json',capture)
        result=export_manifest(cp,path)
        self.assertEqual(result['tensors'][0]['coordinates'],spec['tensors'][0]['coordinates'])
        a['tensors'][0]['coordinates']['tokens'][1]=True;c.write('inventory.json',a['tensors'])
        with self.assertRaisesRegex(ValueError,'coordinates'):export_manifest(cp,path)
        a['tensors'][0]['coordinates']['tokens']=[0,1]
        a['tensors'][0]['coordinates']['step']=1;c.write('inventory.json',a['tensors'])
        with self.assertRaisesRegex(ValueError,'coordinates'):export_manifest(cp,path)


class MsProbePlanTests(unittest.TestCase):
    def model(self):
        class Parameter:
            def numel(self):return 4
            def element_size(self):return 2
        class Block:
            def parameters(self,recurse=False):
                assert recurse is False
                return []
        class Norm:
            def parameters(self,recurse=False):
                assert recurse is False
                return [Parameter()]
        class Model:
            def named_modules(self):
                return iter([('layers.2',Block()),('layers.20',Block()),('layers.2.norm',Norm())])
        return Model()

    def test_full_selectors_and_direct_parameter_bytes(self):
        from msprobe_capture import plan_modules
        result=plan_modules(self.model(),['layers.2','layers.2.norm'],
                            {'layers.2':64,'layers.2.norm':32},'dump',rank=3,step=2,max_tensor_bytes=104)
        self.assertEqual(result['tensor_bytes'],104)
        self.assertEqual(result['observations'][1]['parameter_bytes'],8)
        self.assertEqual(result['config']['tensor']['list'],
                         ['Module.layers.2.Block.forward.0','Module.layers.2.norm.Norm.forward.0'])
        self.assertEqual(result['config']['rank'],[3]);self.assertEqual(result['config']['step'],[2])
        json.dumps(result,allow_nan=False)

    def test_budget_and_unknown_or_unbounded_selection(self):
        from msprobe_capture import plan_modules
        for paths,estimates,budget in [([],{},100),(['missing'],{'missing':1},100),
            (['layers.2'],{},100),(['layers.2'],{'layers.2':True},100),
            (['layers.2.norm'],{'layers.2.norm':32},39),
            (['layers.2','layers.2'],{'layers.2':1},100)]:
            with self.subTest(paths=paths,estimates=estimates),self.assertRaises(ValueError):
                plan_modules(self.model(),paths,estimates,'dump',max_tensor_bytes=budget)
