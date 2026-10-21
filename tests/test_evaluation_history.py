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
        self.data = json.loads((ROOT/'site/data/evaluations.json').read_text())

    def check_data(self, mutate, valid=False):
        with tempfile.TemporaryDirectory() as tmp:
            site=Path(tmp);shutil.copytree(ROOT/'site',site,dirs_exist_ok=True)
            data=copy.deepcopy(self.data);mutate(data)
            (site/'data/evaluations.json').write_text(json.dumps(data))
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
    def evaluate(self, expression):
        script="const h=require('./site/history.js');const d=require('./site/data/evaluations.json');d.evaluations=d.evaluations.filter(r=>['trl-sfttrainer-checkpoint-20261021','trl-sfttrainer-lora-20261021','torch-bf16-fp16-20261021'].includes(r.id));console.log(JSON.stringify("+expression+"))"
        result=subprocess.run(['node','-e',script],cwd=ROOT,capture_output=True,text=True,check=True)
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

    def test_routes_and_legacy_repair_links(self):
        self.assertEqual(self.evaluate("h.routeForHash('#evaluations')"),{'kind':'evaluations','id':'evaluations'})
        self.assertEqual(self.evaluate("h.routeForHash('#evaluation-trl-checkpoint')")['kind'],'evaluations')
        for prefix in ('repair','fix','library'):
            self.assertEqual(self.evaluate("h.routeForHash('#"+prefix+"-existing')")['kind'],'repairs')
        self.assertIsNone(self.evaluate("h.routeForHash('#%E0%A4%A')"))
        self.assertIsNone(self.evaluate("h.routeForHash('#unrelated')"))


if __name__=='__main__':unittest.main()
