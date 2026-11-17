"""Public module boundaries and standalone skill compatibility."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    def isolated(self, program, *args, cwd=None):
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', program,
                                 str(ROOT), *map(str, args)], cwd=cwd,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_rsi_uses_core_without_path_mutation_or_bare_module_collision(self):
        self.isolated("""
import sys,types
sys.path.insert(0,sys.argv[1])
foreign=types.ModuleType('workflow');sys.modules['workflow']=foreign
before=sys.path.copy()
from precision_rsi import loop,memory,harness
from precision_core import workflow
assert loop.run_workflow is workflow.run_workflow
assert memory.fingerprint is workflow.fingerprint
assert sys.path==before
assert sys.modules['workflow'] is foreign
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
""")

    def test_exported_skill_cli_is_independent_of_checkout_and_pythonpath(self):
        from precision_maintenance.install import install_skills
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            install_skills(ROOT/'skills',root/'skills')
            scripts=root/'skills/precision-compare/scripts'
            result=subprocess.run([sys.executable,'-I','-S','-B',str(scripts/'record_run.py'),
                '--output-root',str(root/'runs'),'--side','cpu','--task','portable',
                '--timeout','5','--',sys.executable,'-c','print("portable");raise SystemExit(3)'],
                cwd=root,capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,3,result.stderr)
            receipt=json.loads(result.stdout)
            self.assertEqual(receipt['status'],'failed')
            self.assertEqual(Path(receipt['run'],'stdout.bin').read_text(),'portable\n')
            for tool in ('workflow','resource_catalog','evidence_bundle','runtime_probe'):
                checked=subprocess.run([sys.executable,'-I','-S','-B',str(scripts/(tool+'.py')),
                    '--help'],cwd=root,capture_output=True,text=True,timeout=10)
                self.assertEqual(checked.returncode,0,checked.stderr)
            help_result=subprocess.run([sys.executable,'-I','-S','-B',str(scripts/'tool_help.py'),
                '--full','run_workflow'],cwd=root,capture_output=True,text=True,timeout=10)
            self.assertEqual(help_result.returncode,0,help_result.stderr)
            self.assertIn('run_workflow(',help_result.stdout)

    def test_core_and_legacy_imports_share_one_implementation(self):
        self.isolated("""
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from precision_core import workflow as canonical
sys.path.insert(0,str(Path(sys.argv[1])/'skills/precision-compare/scripts'))
import workflow
assert workflow is canonical
assert workflow.ENGINE_FILES[0].parent.name=='precision_core'
""")

    def test_community_index_runs_without_rsi_or_model_dependencies(self):
        self.isolated("""
import sys
sys.path.insert(0,sys.argv[1])
before=sys.path.copy()
from community.records.catalog import Catalog
catalog=Catalog(sys.argv[1]+'/community/site')
assert len(catalog.find_comparisons())==130
assert not any(x=='precision_rsi' or x.startswith('precision_rsi.') for x in sys.modules)
assert 'precision_core' not in sys.modules
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
assert sys.path==before
assert catalog.related('comparison',catalog.find_comparisons()[0]['id'])==[]
""")

    def test_dispatcher_and_canonical_module_commands_agree(self):
        for module, command in (('community.records.validation', 'validate-history'),
                                ('community.records.differences', 'validate-differences'),
                                ('precision_maintenance.export_statistics', 'export-statistics')):
            args = ['--help'] if command == 'export-statistics' else []
            direct = subprocess.run([sys.executable, '-B', '-m', module, *args],
                cwd=ROOT, capture_output=True, text=True, timeout=15)
            dispatched = subprocess.run([sys.executable, '-B', '-m', 'precision_maintenance', command, *args],
                cwd=ROOT, capture_output=True, text=True, timeout=15)
            self.assertEqual(direct.returncode, 0, direct.stderr)
            self.assertEqual(dispatched.returncode, 0, dispatched.stderr)
            if args:
                self.assertIn('--source', direct.stdout)
                self.assertIn('--source', dispatched.stdout)
            else:
                self.assertEqual(direct.stdout, dispatched.stdout)


class CommunityRelationTests(unittest.TestCase):
    def setUp(self):
        import hashlib
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.site=Path(self.temp.name)
        (self.site/'data').mkdir()
        (self.site/'evidence').mkdir()
        proof=self.site/'evidence/proof.json'
        proof.write_text('{"scope":"fixture workload, one update"}\n')
        self.proof=[dict(file='evidence/proof.json',sha256=hashlib.sha256(proof.read_bytes()).hexdigest())]
        self.evaluation=dict(id='same-id',model=dict(name='model'),task='rl',acceptance=None,sides=[
            dict(hardware='Card A',libraries=[dict(name='Library A',version='1')]),
            dict(hardware='Card B',libraries=[dict(name='Library B',version='2')])])
        self.write('evaluations',dict(schema_version=1,evaluations=[self.evaluation]))
        self.write('repairs',dict(schema_version=1,repairs=[dict(id='same-id')]))
        self.write('differences',dict(schema_version=1,records=[]))
        self.graph=dict(schema_version=1,entities=[],links=[])

    def write(self,name,data):
        (self.site/'data'/f'{name}.json').write_text(json.dumps(data))

    def catalog(self):
        from community.records.catalog import Catalog
        self.write('relations',self.graph)
        return Catalog(self.site)

    def chain(self):
        for kind in ('investigation','issue','candidate','verification','workload'):
            self.graph['entities'].append(dict(id=kind,kind=kind,scope='fixture workload, one update',evidence=self.proof))
        for relation,left,right in (('investigates','investigation','comparison'),
                                   ('confirms','investigation','issue'),
                                   ('addresses','candidate','issue'),
                                   ('verifies','verification','candidate'),
                                   ('uses_workload','verification','workload')):
            self.graph['links'].append(dict(id=relation.replace('_','-'),relation=relation,
                **{'from':dict(kind=left,id=left),'to':dict(kind=right,id=right if right!='comparison' else 'same-id')},
                scope='fixture workload, one update',evidence=self.proof))

    def test_records_preserved_and_relations_never_inferred_from_ids(self):
        before=(self.site/'data/evaluations.json').read_bytes()
        catalog=self.catalog()
        self.assertEqual(catalog.related('comparison','same-id'),[])
        self.assertEqual(catalog.get('comparison','same-id')['record'],self.evaluation)
        self.assertEqual(catalog.get('repair','same-id')['kind'],'repair')
        record=catalog.get('comparison','same-id')
        record['record']['acceptance']=True
        self.assertIsNone(catalog.get('comparison','same-id')['record']['acceptance'])
        self.assertEqual(before,(self.site/'data/evaluations.json').read_bytes())

    def test_hardware_library_and_version_match_the_same_side(self):
        catalog=self.catalog()
        self.assertEqual(catalog.find_comparisons(hardware='Card A',library=dict(name='Library B')),[])
        self.assertEqual(len(catalog.find_comparisons(hardware='Card B',library=dict(name='Library B',version='2'))),1)
        self.assertEqual(catalog.find_comparisons(library=dict(name='Library B',version='1')),[])
        self.assertEqual(catalog.find_comparisons(model='different'),[])
        self.assertEqual(len(catalog.find_comparisons(task='rl')),1)

    def test_explicit_chain_and_source_hashes_survive_detached_reads(self):
        import hashlib
        self.chain()
        catalog=self.catalog()
        self.assertEqual(len(catalog.related('issue','issue')),2)
        entry=catalog.get('verification','verification')
        self.assertEqual(entry['source']['pointer'],'/entities/3')
        self.assertEqual(entry['source']['sha256'],hashlib.sha256((self.site/'data/relations.json').read_bytes()).hexdigest())
        catalog.related('issue','issue')[0]['scope']='changed'
        self.assertEqual(catalog.related('issue','issue')[0]['scope'],'fixture workload, one update')

    def test_dangling_duplicate_and_wrong_kind_links_are_rejected(self):
        import copy
        self.chain()
        good=copy.deepcopy(self.graph)
        cases=[]
        value=copy.deepcopy(good);value['links'][0]['to']['id']='missing';cases.append(value)
        value=copy.deepcopy(good);value['links'][0]['to']=dict(kind='repair',id='same-id');cases.append(value)
        value=copy.deepcopy(good);value['links'].append(copy.deepcopy(value['links'][0]));cases.append(value)
        value=copy.deepcopy(good);value['links'].append(dict(value['links'][0],id='other'));cases.append(value)
        value=copy.deepcopy(good);value['entities'].append(dict(value['entities'][0],kind='issue'));cases.append(value)
        for graph in cases:
            with self.subTest(graph=graph):
                self.graph=graph
                with self.assertRaises(ValueError): self.catalog()

    def test_verification_requires_candidate_workload_scope_and_evidence(self):
        import copy
        self.chain()
        good=copy.deepcopy(self.graph)
        for relation in ('verifies','uses_workload'):
            self.graph=copy.deepcopy(good)
            self.graph['links']=[x for x in self.graph['links'] if x['relation']!=relation]
            with self.assertRaisesRegex(ValueError,'original workload'): self.catalog()
        for field,value in (('scope',''),('evidence',[])):
            self.graph=copy.deepcopy(good)
            self.graph['links'][0][field]=value
            with self.assertRaises(ValueError): self.catalog()

    def test_changed_or_escaping_evidence_and_nonfinite_json_rejected(self):
        self.chain()
        self.graph['links'][0]['evidence']=[dict(file='../outside',sha256='a'*64)]
        with self.assertRaises(ValueError): self.catalog()
        self.graph['links'][0]['evidence']=self.proof
        (self.site/'evidence/proof.json').write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'): self.catalog()
        from community.records.catalog import Catalog
        for text in ('{"schema_version":1,"records":[],"records":[]}',
                     '{"schema_version":1,"records":[{"id":"x","metric":1e999}]}'):
            (self.site/'data/differences.json').write_text(text)
            with self.assertRaises(ValueError): Catalog(self.site)

    def test_relation_history_is_append_only(self):
        from precision_maintenance.validate_immutability import COLLECTIONS,compare_documents
        self.assertEqual(COLLECTIONS['community/site/data/relations.json'],('entities','links'))
        self.chain()
        import copy
        old=copy.deepcopy(self.graph)
        self.graph['links'][0]['scope']='rewritten'
        self.assertTrue(compare_documents('relations.json',old,self.graph,('entities','links')))

    def test_index_hash_is_bound_to_parsed_bytes_during_source_replacement(self):
        import hashlib
        from unittest.mock import patch
        from community.records.catalog import Catalog
        target=(self.site/'data/evaluations.json').resolve()
        original=target.read_bytes()
        read_bytes,read_text=Path.read_bytes,Path.read_text
        replacements=[]
        def replace(path):
            if path==target:
                replacements.append(True)
                self.write('evaluations',dict(schema_version=1,evaluations=[dict(self.evaluation,id='replacement')]))
        def bytes_read(path,*args,**kwargs):
            value=read_bytes(path,*args,**kwargs)
            replace(path)
            return value
        def text_read(path,*args,**kwargs):
            value=read_text(path,*args,**kwargs)
            replace(path)
            return value
        with patch.object(Path,'read_bytes',bytes_read),patch.object(Path,'read_text',text_read):
            catalog=Catalog(self.site)
        self.assertTrue(replacements)
        self.assertEqual(catalog.get('comparison','same-id')['source']['sha256'],
                         hashlib.sha256(original).hexdigest())
        self.assertNotEqual(target.read_bytes(),original)
