"""Regressions for published repair records and evidence boundaries."""
import copy
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('validate_history', ROOT / '.github/validate_history.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RepairHistoryTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / 'site/data/repairs.json').read_text(encoding='utf-8'))

    def check_invalid(self, mutate):
        with tempfile.TemporaryDirectory() as temp:
            site = Path(temp)
            (site / 'data').mkdir()
            data = copy.deepcopy(self.data)
            # Supply all valid local evidence before mutating the record.
            for repair in data['repairs']:
                for item in repair['evidence']:
                    path = site / item['url']
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('evidence')
            mutate(data)
            (site / 'data/repairs.json').write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                MODULE.validate(site)

    def test_published_history(self):
        self.assertGreaterEqual(MODULE.validate(ROOT / 'site'), 1)

    def test_duplicate_identity(self):
        self.check_invalid(lambda d: d['repairs'].append(copy.deepcopy(d['repairs'][0])))

    def test_invalid_coverage(self):
        self.check_invalid(lambda d: d['repairs'][0]['validation']['coverage'].update(observed=33, expected=32))

    def test_nonfinite_metric(self):
        def nonfinite(data):
            data['repairs'][0]['validation']['metrics'] = [{
                'id': 'invalid-error', 'label': 'Error', 'unit': 'value',
                'before': 0, 'after': float('nan')}]
        self.check_invalid(nonfinite)

    def test_escaped_evidence(self):
        self.check_invalid(lambda d: d['repairs'][0]['evidence'][0].update(url='../README.md'))

    def test_missing_parent_repair(self):
        self.check_invalid(lambda d: d['repairs'][0].update(related_repair_id='missing-repair'))

    def test_cyclic_repair_relationship(self):
        def cycle(data):
            root = next(r for r in data['repairs'] if not r.get('related_repair_id'))
            child = next(r for r in data['repairs'] if r.get('related_repair_id'))
            root['related_repair_id'] = child['id']
        self.check_invalid(cycle)

    def test_missing_library_group(self):
        self.check_invalid(lambda d: d.update(library_groups=[]))

    def test_conflicting_followup_group(self):
        self.check_invalid(lambda d: d['repairs'][0].update(library_group_id='other-library'))

    def test_affected_library_belongs_to_group(self):
        def unrelated(data):
            root = next(r for r in data['repairs'] if not r.get('related_repair_id'))
            root['affected_library'] = 'unrelated-library'
        self.check_invalid(unrelated)

    def test_unsafe_evidence_scheme(self):
        self.check_invalid(lambda d: d['repairs'][0]['evidence'][0].update(url='javascript:alert(1)'))


@unittest.skipUnless(shutil.which('node'), 'Node.js is needed for browser data-model regressions')
class HistoryGroupingTests(unittest.TestCase):
    def evaluate(self, data, expression):
        script = "const h=require('./site/history.js');const d=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(" + expression + "));"
        result = subprocess.run(['node', '-e', script], input=json.dumps(data), cwd=ROOT,
                                text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def data(self):
        environment = {'libraries': [{'name': 'MindSpeed'}, {'name': 'Megatron Bridge'}], 'hardware': 'fixture-hardware'}
        root = {'id': 'fixture-fix', 'library_group_id': 'fixture-group', 'title': 'RMSNorm repair',
                'date': '2026-10-21', 'cause': {'summary': 'gamma'}, 'environment': environment,
                'model': {'name': 'fixture-model'}, 'verification_title': 'forward'}
        followup = copy.deepcopy(root)
        followup.update(id='fixture-followup', related_repair_id=root['id'], verification_title='backward')
        return {'library_groups': [{'id': 'fixture-group', 'title': 'Bridge ↔ MindSpeed',
                                   'components': ['Megatron Bridge', 'MindSpeed']}],
                'repairs': [followup, root]}

    def test_followup_is_a_verification_of_one_fix(self):
        counts = self.evaluate(self.data(), 'h.counts(h.buildCatalog(d))')
        self.assertEqual(counts, {'groups': 1, 'fixes': 1, 'verifications': 2})

    def test_multiple_fixes_and_library_groups(self):
        data = self.data()
        root = copy.deepcopy(next(r for r in data['repairs'] if not r.get('related_repair_id')))
        root['id'] = 'fixture-second-fix'
        data['repairs'].append(root)
        self.assertEqual(self.evaluate(data, 'h.counts(h.buildCatalog(d))'),
                         {'groups': 1, 'fixes': 2, 'verifications': 3})
        other = copy.deepcopy(root)
        other.update(id='fixture-other-library', library_group_id='fixture-torch')
        data['library_groups'].append({'id': 'fixture-torch', 'title': 'Torch', 'components': ['Torch']})
        data['repairs'].append(other)
        self.assertEqual(self.evaluate(data, 'h.counts(h.buildCatalog(d))'),
                         {'groups': 2, 'fixes': 3, 'verifications': 4})

    def test_filters_keep_group_fix_and_verification_hierarchy(self):
        data = self.data()
        self.assertEqual(self.evaluate(data, "h.counts(h.filterCatalog(h.buildCatalog(d),{library:'MindSpeed'}))"),
                         {'groups': 1, 'fixes': 1, 'verifications': 2})
        self.assertEqual(self.evaluate(data, "h.counts(h.filterCatalog(h.buildCatalog(d),{library:'Torch'}))"),
                         {'groups': 0, 'fixes': 0, 'verifications': 0})
        title = data['repairs'][0]['verification_title']
        self.assertEqual(self.evaluate(data, 'h.counts(h.filterCatalog(h.buildCatalog(d),{query:' + json.dumps(title) + '}))'),
                         {'groups': 1, 'fixes': 1, 'verifications': 1})

    def test_orphan_followup_is_rejected(self):
        data = self.data()
        data['repairs'][0]['related_repair_id'] = 'missing-fix'
        self.assertTrue(self.evaluate(data, "(()=>{try{h.buildCatalog(d);return false}catch{return true}})()"))

    def test_cycle_and_conflicting_group_are_rejected(self):
        data = self.data()
        root = next(r for r in data['repairs'] if not r.get('related_repair_id'))
        root['related_repair_id'] = data['repairs'][0]['id']
        self.assertTrue(self.evaluate(data, "(()=>{try{h.buildCatalog(d);return false}catch{return true}})()"))
        data = self.data()
        data['repairs'][0]['library_group_id'] = 'conflicting-group'
        self.assertTrue(self.evaluate(data, "(()=>{try{h.buildCatalog(d);return false}catch{return true}})()"))


@unittest.skipUnless(shutil.which('node'), 'Node.js is needed for repair presentation regressions')
class RepairPresentationTests(unittest.TestCase):
    def evaluate(self, expression):
        script = "const p=require('./site/repair-presentation.js');const h=require('./site/history.js');const d=require('./site/data/repairs.json');console.log(JSON.stringify(" + expression + "));"
        return json.loads(subprocess.check_output(['node', '-e', script], cwd=ROOT, text=True))

    def test_every_published_fix_has_workflow_and_version(self):
        rows = self.evaluate('h.buildCatalog(d).flatMap(g=>g.fixes.map(f=>p.overview(f)))')
        self.assertTrue(all(row['tasks'] and row['version'] != '版本未记录' for row in rows))
        self.assertEqual({task for row in rows for task in row['tasks']}, {'inference', 'sft', 'rl'})

    def test_root_result_is_not_replaced_by_later_optimization(self):
        row = self.evaluate("(()=>{const f=h.buildCatalog(d).flatMap(g=>g.fixes).find(f=>f.records.length>1);return {actual:p.overview(f).metric,original:f.root.validation.metrics[0]}})()")
        self.assertEqual(row['actual'], row['original'])

    def test_task_filter_preserves_records_and_shared_training(self):
        for task in ('inference', 'sft', 'rl'):
            rows = self.evaluate("p.filterTask(h.buildCatalog(d)," + json.dumps(task) + ").flatMap(g=>g.fixes.map(f=>({tasks:p.overview(f).tasks,records:f.records.length})))")
            self.assertTrue(rows)
            self.assertTrue(all(task in row['tasks'] and row['records'] for row in rows))
        shared = self.evaluate("p.overview(h.buildCatalog(d).flatMap(g=>g.fixes).find(f=>f.root.id.includes('radam')))")
        self.assertEqual(shared['tasks'], ['sft', 'rl'])
        self.assertTrue(shared['sharedTraining'])

    def test_new_record_metadata_and_unknown_workflow(self):
        row = self.evaluate("p.overview({root:{id:'new',title:'Fallback',affected_library:'NewLib',environment:{libraries:[{name:'NewLib',version:'1.2'}]},presentation:{title:'New module',tasks:['rl','rl','unknown']},validation:{metrics:[]}}})")
        self.assertEqual(row['tasks'], ['rl'])
        self.assertEqual(row['title'], 'New module')
        self.assertIsNone(row['metric'])
        unknown = self.evaluate("p.overview({root:{id:'unknown',title:'Unknown'}})")
        self.assertEqual(unknown['tasks'], [])

    def test_compact_values_keep_zero_and_tiny_distinct(self):
        values = self.evaluate('[0,8.881784197001252e-16,496,0.026155292987823486].map(p.compactNumber)')
        self.assertEqual(values, ['0', '8.88e-16', '496', '0.02616'])
