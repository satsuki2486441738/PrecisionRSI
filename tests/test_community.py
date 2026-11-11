"""Community projections preserve source scope and side-specific environment joins."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]

@unittest.skipUnless(shutil.which('node'), 'Node.js required')
class CommunityTests(unittest.TestCase):
    def run_js(self, expression):
        script = """
        global.PrecisionHistory=require('./site/history.js');
        global.PrecisionRepairPresentation=require('./site/repair-presentation.js');
        const api=require('./site/community.js');
        const rd=require('./site/data/repairs.json'), ed=require('./site/data/evaluations.json');
        const before=JSON.stringify([rd,ed]);const rows=api.build(rd,ed);
        console.log(JSON.stringify(""" + expression + '));'
        return json.loads(subprocess.run(['node', '-e', script], cwd=ROOT, capture_output=True, text=True, check=True).stdout)

    def test_actual_sources_are_immutable_and_roots_are_not_duplicated(self):
        result = self.run_js("""({unchanged:before===JSON.stringify([rd,ed]),
          fixes:rows.filter(r=>r.kind==='repair').map(r=>r.id),
          expected:rd.repairs.filter(r=>!r.related_repair_id).map(r=>r.id),
          evals:rows.filter(r=>r.kind==='evaluation').map(r=>r.id),
          public:ed.evaluations.filter(r=>r.visibility!=='archived').map(r=>r.id)})""")
        self.assertTrue(result['unchanged'])
        self.assertCountEqual(result['fixes'], result['expected'])
        self.assertCountEqual(result['evals'], result['public'])

    def test_no_cross_side_hardware_library_join(self):
        result = self.run_js("""(() => {
          const r={id:'fixture',task:'inference',configs:[
            {hardware:['Device A'],libraries:[{name:'GPU library'}],model:'M'},
            {hardware:['Device B'],libraries:[{name:'NPU library'}],model:'M'}]};
          return [api.filter([r],{hardware:'Device A',library:'NPU library'}).length,
            api.filter([r],{hardware:'Device B',library:'NPU library'}).length];
        })()""")
        self.assertEqual(result, [0, 1])

    def test_hardware_descriptions_and_future_ecosystems(self):
        result = self.run_js("""[api.hardware('CPU · 单线程 · FP32 模型 / FP64 helper'),
          api.hardware('Custom FPGA generation Q'),api.libraryKey('torch'),api.libraryKey('vllm'),api.libraryKey('torch-npu')]""")
        self.assertEqual(result, [['CPU'], ['Custom FPGA generation Q'], 'PyTorch', 'vLLM', 'torch_npu'])

    def test_repair_dependency_is_not_attributed_as_defect(self):
        result = self.run_js("""(() => {
          const r=rows.find(r=>r.id==='trl-0710-ppo-whitening-tail-leakage');
          return [api.filter([r],{library:'TRL'}).length,api.filter([r],{library:'PyTorch'}).length];
        })()""")
        self.assertEqual(result, [1, 0])

    def test_all_actual_hardware_and_library_filters_have_real_config(self):
        result = self.run_js("""rows.every(r=>r.configs.every(c=>c.hardware.every(h=>
          api.indexedLibraries(c).every(l=>api.filter([r],{hardware:h,library:l}).length===1))))""")
        self.assertTrue(result)

    def test_investigations_acceptance_and_links_preserved(self):
        result = self.run_js("""({pending:rows.filter(r=>r.kind==='repair'&&r.source.status==='investigating').length,
          acceptancePreserved:rows.every(r=>r.kind!=='evaluation'||r.source===ed.evaluations.find(x=>x.id===r.id)),
          links:rows.every(r=>r.url.startsWith('./history.html#'))})""")
        self.assertGreater(result['pending'], 0)
        self.assertTrue(result['acceptancePreserved'])
        self.assertTrue(result['links'])
