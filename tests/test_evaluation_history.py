"""Evaluation history validates peer semantics and keeps repair routing intact."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('evaluation_validator', ROOT / '.github/validate_history.py')
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class EvaluationDataTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT/'site/data/evaluations.json').read_text(encoding='utf-8'))

    def check_data(self, mutate, valid=False):
        with tempfile.TemporaryDirectory() as tmp:
            site=Path(tmp);shutil.copytree(ROOT/'site',site,dirs_exist_ok=True)
            data=copy.deepcopy(self.data);mutate(data)
            (site/'data/evaluations.json').write_text(json.dumps(data),encoding='utf-8')
            if valid:
                self.assertEqual(VALIDATOR.validate_evaluations(site),len(data['evaluations']))
            else:
                with self.assertRaises(ValueError):VALIDATOR.validate_evaluations(site)

    def test_published_data(self):
        self.assertEqual(VALIDATOR.validate_evaluations(ROOT/'site'),len(self.data['evaluations']))
        self.assertGreater(len(self.data['evaluations']),0)

    def test_duplicate_record_and_side_identities(self):
        self.check_data(lambda d:d['evaluations'].append(copy.deepcopy(d['evaluations'][0])))
        self.check_data(lambda d:d['evaluations'][0]['sides'].append(copy.deepcopy(d['evaluations'][0]['sides'][0])))

    def test_pairs_need_two_declared_distinct_sides(self):
        self.check_data(lambda d:d['evaluations'][0]['comparisons'][0].update(side_ids=['missing','missing']))
        self.check_data(lambda d:d['evaluations'][0]['comparisons'][0].update(side_ids=['gpu-continuous','missing']))

    def test_side_metrics_must_belong_to_pair(self):
        self.check_data(lambda d:d['evaluations'][0]['comparisons'][0]['metrics'][0].update(side_values={'wrong':1}))

    def test_metric_values_are_finite_and_zero_is_valid(self):
        self.check_data(lambda d:d['evaluations'][0]['comparisons'][0]['metrics'][0].update(value=float('nan')))
        self.check_data(lambda d:d['evaluations'][0]['comparisons'][0]['metrics'][0].update(value=0),valid=True)
        self.check_data(lambda d:d['evaluations'][0]['comparisons'][0]['metrics'][1].update(value=-.25),valid=True)

    def test_coverage_is_observation_count_not_equal_array_count(self):
        record=self.data['evaluations'][0]
        pair=next(c for c in record['comparisons'] if c['id']=='npu-continuation')
        exact=next(m for m in pair['metrics'] if m['id']=='exact-arrays')
        self.assertEqual(exact['value'],298)
        coverage=next(c for c in pair['coverage'] if c['unit']=='arrays')
        self.assertEqual((coverage['observed'],coverage['expected']),(448,448))
        self.check_data(lambda d:d['evaluations'][0]['comparisons'][0]['coverage'][0].update(observed=21))

    def test_explicit_numerical_acceptance_is_independent_of_completion(self):
        self.check_data(lambda d:d['evaluations'][0].update(acceptance={'status':'accepted'}),valid=True)

    def test_archive_and_evidence(self):
        self.check_data(lambda d:d['evaluations'][0]['archive'].update(revision='master'))
        self.check_data(lambda d:d['evaluations'][0]['evidence'][0].update(url='../README.md'))
        self.check_data(lambda d:d['evaluations'][0]['evidence'][0].update(url='javascript:alert(1)'))


@unittest.skipUnless(shutil.which('node'),'Node.js is needed for frontend model tests')
class EvaluationModelTests(unittest.TestCase):
    def evaluate(self, expression, all_records=False):
        script="const h=require('./site/history.js');const d=require('./site/data/evaluations.json');"
        if not all_records:
            script+="d.evaluations=d.evaluations.filter(r=>['trl-sfttrainer-checkpoint-20261021','trl-sfttrainer-lora-20261021','torch-bf16-fp16-20261021'].includes(r.id));"
        script+="console.log(JSON.stringify("+expression+"))"
        result=subprocess.run(['node','-e',script],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True)
        return json.loads(result.stdout)

    def test_counts_and_task_groups(self):
        self.assertEqual(self.evaluate('h.evaluationCounts(d.evaluations)'),{'experiments':3,'comparisons':7,'hardware':2})
        self.assertEqual(self.evaluate('h.evaluationGroups(d.evaluations).map(g=>g.id)'),['inference','sft'])

    def test_filters_preserve_whole_peer_comparisons(self):
        self.assertEqual(self.evaluate("h.filterEvaluations(d.evaluations,{hardware:'Ascend 910C'}).length"),2)
        self.assertEqual(self.evaluate("h.filterEvaluations(d.evaluations,{library:'TRL',task:'inference'}).length"),0)
        self.assertEqual(self.evaluate("h.filterEvaluations(d.evaluations,{query:'FP16'}).length"),1)
        self.assertEqual(self.evaluate("h.filterEvaluations(d.evaluations,{task:'rl'}).length"),0)
        self.assertEqual(self.evaluate("h.filterEvaluations(d.evaluations,{hardware:'Ascend 910C'})[0].sides.length"),4)

    def test_published_facets_group_model_aliases_and_hardware_annotations(self):
        facets=self.evaluate('h.evaluationFacets(d.evaluations)',all_records=True)
        self.assertEqual(set(facets['models']),{
            'Qwen2.5-0.5B-Instruct','Qwen3-0.6B','Qwen3-0.6B-Base','Qwen3.5-9B-Base',
            'SmolLM2-135M-Instruct','tiny Llama','tiny Megatron GPT','tiny Qwen2','tiny Qwen3'})
        self.assertEqual(set(facets['hardware']),{
            'CPU','Ascend 910','Ascend 910C','Ascend NPU',
            'NVIDIA GeForce RTX 5060','NVIDIA H100','NVIDIA H200'})
        self.assertEqual(facets,self.evaluate("(()=>{const f=h.evaluationFacets(d.evaluations);return {models:[...f.models].sort((a,b)=>a.localeCompare(b,'en')),hardware:[...f.hardware].sort((a,b)=>a.localeCompare(b,'en'))}})()",all_records=True))
        self.assertEqual(self.evaluate('h.evaluationCounts(d.evaluations).hardware',all_records=True),len(facets['hardware']))

    def test_canonical_filters_reach_every_published_alias_without_changing_records(self):
        result=self.evaluate("""(()=>{
          const before=JSON.stringify(d.evaluations),facets=h.evaluationFacets(d.evaluations);
          const modelRecords=facets.models.flatMap(model=>h.filterEvaluations(d.evaluations,{model}));
          const cpuRecords=d.evaluations.filter(r=>r.sides.some(s=>/\\bCPU\\b/i.test(s.hardware)));
          const tinyQwen2=d.evaluations.filter(r=>r.model.name.includes('tiny Qwen2')||r.model.name.startsWith('Qwen2 随机'));
          return {unchanged:before===JSON.stringify(d.evaluations),
            sameObjects:modelRecords.every(r=>d.evaluations.includes(r)),
            modelIds:modelRecords.map(r=>r.id).sort(),allIds:d.evaluations.map(r=>r.id).sort(),
            cpuIds:h.filterEvaluations(d.evaluations,{hardware:'CPU'}).map(r=>r.id).sort(),expectedCpuIds:cpuRecords.map(r=>r.id).sort(),
            tinyIds:h.filterEvaluations(d.evaluations,{model:'tiny Qwen2'}).map(r=>r.id).sort(),expectedTinyIds:tinyQwen2.map(r=>r.id).sort()};
        })()""",all_records=True)
        self.assertTrue(result['unchanged'])
        self.assertTrue(result['sameObjects'])
        self.assertEqual(result['modelIds'],result['allIds'])
        self.assertEqual(result['cpuIds'],result['expectedCpuIds'])
        self.assertEqual(result['tinyIds'],result['expectedTinyIds'])

    def test_model_seed_metadata_does_not_merge_real_model_sizes(self):
        aliases={
            'tiny Qwen2（seed 20261019）':'tiny Qwen2',
            'tiny Qwen2（seed 20261020）':'tiny Qwen2',
            'Qwen2 随机 45,376 参数（1 layer / hidden64 / vocab64）':'tiny Qwen2',
            '随机 tiny Qwen3 · untied':'tiny Qwen3',
            'tiny Qwen3（种子20261022随机权重）':'tiny Qwen3',
            'Qwen/Qwen3-0.6B · untied 派生配置':'Qwen3-0.6B',
            'Qwen/Qwen3-0.6B (seed20261019)':'Qwen3-0.6B',
            'Qwen/Qwen3-0.6B (seed 20261020)':'Qwen3-0.6B',
            'Qwen/Qwen3-0.6B-Base':'Qwen3-0.6B-Base',
            'Qwen/Qwen3-1.7B':'Qwen3-1.7B',
            'Qwen/Qwen2.5-0.5B-Instruct':'Qwen2.5-0.5B-Instruct',
            'Qwen/Qwen2.5-1.5B-Instruct':'Qwen2.5-1.5B-Instruct',
            'SmolLM2-360M-Instruct':'SmolLM2-360M-Instruct'}
        result=self.evaluate('Object.fromEntries('+json.dumps(list(aliases))+'.map(name=>[name,h.evaluationModel({model:{name}})]))')
        self.assertEqual(result,aliases)

    def test_mixed_hardware_keeps_each_platform_and_ignores_device_counts(self):
        cases=[
            ('2 × NVIDIA H100 80GB HBM3（训练 1 卡 + rollout 1 卡） + Ascend 910B / npu:0 + Ascend910C · physical8; CPU（型号未记录）',
             ['NVIDIA H100','Ascend 910B','Ascend 910C','CPU']),
            ('NVIDIA GeForce RTX5060 8151MiB / NVIDIA RTX 5060 · 8GB · WSL Ubuntu-24.04 / NVIDIA GeForce RTX 4090',
             ['NVIDIA GeForce RTX 5060','NVIDIA GeForce RTX 4090']),
            ('Ascend910_9382 · physical8 / npu:0',['Ascend 910']),
            ('Ascend NPU (device SKU not recorded in this lane)',['Ascend NPU']),
            ('NVIDIA H100 (Ascend910C + RTX5060 + MI300X)',
             ['NVIDIA H100','Ascend 910C','NVIDIA GeForce RTX 5060','AMD Instinct MI300X']),
            ('NVIDIA H100 / GPU 0 / GPU 1 / 4 cards',['NVIDIA H100']),
            ('Ascend 910C npu:0 / 1',['Ascend 910C']),
            ('NVIDIA H100 + NVIDIA H200 + NVIDIA A100 + AMD Instinct MI300X',
             ['NVIDIA H100','NVIDIA H200','NVIDIA A100','AMD Instinct MI300X'])]
        for raw,expected in cases:
            with self.subTest(raw=raw):
                labels=self.evaluate('h.hardwareLabels('+json.dumps(raw)+')')
                self.assertCountEqual(labels,expected)
                self.assertEqual(len(labels),len(set(labels)))

    def test_mixed_side_filters_and_counts_share_the_same_hardware_classes(self):
        result=self.evaluate("""(()=>{
          const record={...d.evaluations[0],sides:[
            {...d.evaluations[0].sides[0],hardware:'2 × NVIDIA H100 + Ascend 910B / npu:0'},
            {...d.evaluations[0].sides[1],hardware:'Ascend910C + NVIDIA H200 + CPU（型号未记录）'},
            {...d.evaluations[0].sides[0],hardware:'NVIDIA H100 / GPU 1'}]};
          const records=[record],hardware=h.evaluationHardware(record);
          return {hardware,counts:h.evaluationCounts(records),facets:h.evaluationFacets(records).hardware,
            matches:hardware.map(label=>h.filterEvaluations(records,{hardware:label})[0]===record),
            missing:h.filterEvaluations(records,{hardware:'Ascend 910'}).length,
            sides:h.filterEvaluations(records,{hardware:'Ascend 910B'})[0].sides.length};
        })()""")
        self.assertCountEqual(result['hardware'],['NVIDIA H100','Ascend 910B','Ascend 910C','NVIDIA H200','CPU'])
        self.assertCountEqual(result['facets'],result['hardware'])
        self.assertEqual(result['counts']['hardware'],5)
        self.assertTrue(all(result['matches']))
        self.assertEqual(result['missing'],0)
        self.assertEqual(result['sides'],3)

    def test_raw_hardware_and_seed_search_remain_available(self):
        for query in ('physical8','npu:0','20261019','134,515,008'):
            with self.subTest(query=query):
                result=self.evaluate("(()=>{const query="+json.dumps(query)+";return {actual:h.filterEvaluations(d.evaluations,{query}).map(r=>r.id).sort(),expected:d.evaluations.filter(r=>JSON.stringify(r).toLocaleLowerCase().includes(query.toLocaleLowerCase())).map(r=>r.id).sort()}})()",all_records=True)
                self.assertTrue(result['expected'])
                self.assertEqual(result['actual'],result['expected'])

    def test_routes_and_legacy_repair_links(self):
        self.assertEqual(self.evaluate("h.routeForHash('#evaluations')"),{'kind':'evaluations','id':'evaluations'})
        self.assertEqual(self.evaluate("h.routeForHash('#evaluation-trl-checkpoint')")['kind'],'evaluations')
        for prefix in ('repair','fix','library'):
            self.assertEqual(self.evaluate("h.routeForHash('#"+prefix+"-existing')")['kind'],'repairs')
        self.assertIsNone(self.evaluate("h.routeForHash('#%E0%A4%A')"))
        self.assertIsNone(self.evaluate("h.routeForHash('#unrelated')"))


if __name__=='__main__':unittest.main()
