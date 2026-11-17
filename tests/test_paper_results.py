from pathlib import Path
import tempfile
import unittest
import json
import zipfile
import subprocess
import sys
import re
import shutil

from precision_maintenance import export_paper_results as m

class PaperExportTests(unittest.TestCase):
    def test_pointer_units_and_nulls(self):
        rows=list(m.numbers({'a/b':{'value':0.003,'unit':'nat/token'},'null':None,'bool':False,'repeat':[1,2]}))
        self.assertEqual(rows[0],('/a~1b/value',.003,'nat/token'))
        self.assertEqual(len(rows),3)
        self.assertEqual(m.unit_for('max_abs',{}),'unspecified; see source context')
    def test_sanitization_preserves_measurements(self):
        data=m.sanitize({'value':.003,'physical_devices':[0,1],'scope':'modelmate-physical0','pin':{'path':'/Users/private/run/result.json','sha256':'abc'},'runtime':'/home/work/user-name/module.py','projects/private/secret.py':'digest','runtime_topology':{'allocation':{'UUID':'private'},'effective_device_environment':{'mask':'private'},'allocated_uuid':'private','attention_mask':[1,1]}})
        self.assertEqual(data['value'],.003)
        self.assertNotIn('physical_devices',data)
        self.assertNotIn('/Users/',json.dumps(data))
        self.assertEqual(data['pin']['sha256'],'abc')
        self.assertNotIn('/home/',json.dumps(data))
        self.assertNotIn('projects/private',json.dumps(data))
        self.assertIn('digest',data.values())
        self.assertEqual(data['runtime_topology'],{'attention_mask':[1,1]})
    def test_public_snapshot_numeric_fields_match_sources(self):
        snapshot=json.loads((m.ROOT/'site/data/paper-results.json').read_text())
        sources={source['id']:source['data'] for source in snapshot['sources']}
        self.assertGreater(len(snapshot['numeric_fields']),0)
        for row in snapshot['numeric_fields']:
            value=sources[row['source_id']]
            for part in row['pointer'].split('/')[1:]:
                key=part.replace('~1','/').replace('~0','~')
                value=value[int(key)] if isinstance(value,list) else value[key]
            self.assertEqual(value,row['value'],(row['source_id'],row['pointer']))

    @unittest.skipUnless(all(path.is_file() for _,_,path in m.registry()),
                         'Full export integration requires the retained private source archives')
    def test_bundle_values_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'export'; result=m.export(out)
            sources={s['id']:s['data'] for s in result['sources']}
            for row in result['numeric_fields']:
                v=sources[row['source_id']]
                for part in row['pointer'].split('/')[1:]:
                    key=part.replace('~1','/').replace('~0','~')
                    v=v[int(key)] if isinstance(v,list) else v[key]
                self.assertEqual(v,row['value'])
            self.assertGreater(len(result['sources']),20)
            self.assertEqual(result['highlights'][0]['value'],298)
            self.assertEqual(result['highlights'][1]['value'],448)
            self.assertAlmostEqual(result['highlights'][8]['value'],0.005357432449655253)
            historical_gap=next(s for s in result['sources'] if s['id']=='deepseek-gap-ledger')
            self.assertIn('HISTORICAL 2026-11-06',historical_gap['title'])
            self.assertTrue(any('Their old gap statements' in note and 'superseded' in note for note in result['notes']))
            self.assertEqual(sources['deepseek41-adapted-core-20261109']['coverage']['adapted_paired'],275)
            self.assertEqual(sources['deepseek41-adapted-core-20261109']['coverage']['original_frozen_paired'],191)
            self.assertFalse(sources['deepseek41-adapted-core-20261109']['whole_task_completed'])
            self.assertEqual(sources['deepseek41-full40-attention-20261109']['summary']['rank0_layer_forward_comparisons'],480)
            head=sources['deepseek41-paired-output-head95-20261109']
            self.assertEqual(head['paired_calls'],95)
            self.assertEqual(head['native_forward_calls'],{'gpu':380,'npu':760})
            self.assertFalse(head['full_backbone_autograd'])
            self.assertFalse(head['whole_task_completed'])
            self.assertTrue(all(not x['before_exact']['A'] and x['before_exact']['B'] for x in head['same_initialization_checks']))
            self.assertNotIn('allocation',head['configurations']['gpu'])
            self.assertIn('deepseek41-paired-output-head95-20261109',m.SUPPLEMENTS)
            http=sources['deepseek41-http128-20261109']
            self.assertFalse(http['whole_task_completed'])
            self.assertEqual(http['coverage']['native_HTTP_requests'],128)
            self.assertEqual(http['coverage']['adapted_HTTP_32768_token_histories'],9)
            self.assertEqual(http['coverage']['ordinary_prefills'],78)
            self.assertEqual(http['coverage']['fresh_teacher_prefills'],32)
            self.assertEqual(http['coverage']['unpaired_candidate_last_distributions'],18)
            self.assertEqual(http['coverage']['planned_target_scalars'],3168)
            self.assertEqual(http['coverage']['candidate_scalar_terms'],24)
            self.assertFalse(http['coverage']['original_default_32K'])
            http_asset=next(a for a in result['assets'] if a['file']=='supplements/deepseek41-http128-20261109.zip')
            self.assertIn('bounded generated full-vocabulary arrays',http_asset['scope'])
            self.assertIn('The HTTP128 supplement ships bounded generated probability arrays',(out/'README.md').read_text())
            key_cache=sources['deepseek41-native-producer-key-cache-20261109']
            self.assertFalse(key_cache['scope']['whole_task_completed'])
            self.assertFalse(key_cache['scope']['same_live_tensor_chain'])
            self.assertEqual(key_cache['scope']['private_full_review']['key_coordinates'],6684672)
            self.assertEqual(key_cache['scope']['portable_replay']['sampled_projection_rows'],256)
            self.assertEqual(key_cache['scope']['portable_replay']['sampled_projection_output_coordinates'],32768)
            self.assertFalse(key_cache['scope']['portable_replay']['full_projection_replay'])
            self.assertEqual(key_cache['scope']['portable_replay']['full_downstream_key_coordinates'],6684672)
            self.assertEqual(key_cache['configuration']['cache_scale_dtype'],'FP16')
            self.assertEqual(key_cache['configuration']['exported_quant_scale_dtype'],'FP32')
            key_asset=next(a for a in result['assets'] if a['file']=='supplements/deepseek41-native-producer-key-cache-20261109.zip')
            self.assertIn('sampled projection inputs and full downstream arrays',key_asset['scope'])
            self.assertIn('FP32-scale reconstruction is a separate metric',key_asset['scope'])
            qli=sources['deepseek41-native-query-key-selection-20261109']
            self.assertEqual(len(m.SUPPLEMENTS),13)
            self.assertEqual(qli['scope']['portable_replay']['source_arrays'],20)
            self.assertEqual(qli['scope']['portable_replay']['actual_numeric_raw_pairs'],74)
            self.assertEqual(qli['scope']['portable_replay']['all_derived_score_coordinates'],208896)
            self.assertEqual(qli['scope']['portable_replay']['finite_derived_score_coordinates'],208876)
            self.assertEqual(qli['scope']['portable_replay']['masked_score_coordinates'],20)
            self.assertFalse(qli['scope']['whole_task_completed'])
            self.assertFalse(qli['scope']['same_live_tensor_chain'])
            self.assertFalse(qli['scope']['full_native_scores_available'])
            self.assertFalse(qli['scope']['candidate_consumer_equivalence'])
            self.assertIsNone(qli['scope']['numerical_acceptance'])
            self.assertEqual(qli['numerics']['selected_top512']['stable_reference_overlap_counts'],{'512':11,'511':4,'510':1})
            self.assertEqual(qli['numerics']['selected_top512']['independent_reference_boundary_inconsistent_queries'],5)
            self.assertFalse(qli['numerics']['selected_top512']['native_topk_defect_inferred'])
            self.assertFalse(qli['numerics']['candidate_mode1']['N2048']['discriminating_top2048_accuracy'])
            self.assertEqual(qli['numerics']['candidate_mode1']['N32768']['stable_reference_overlap'],[2048,2047,2047,2047])
            self.assertEqual(qli['numerics']['native_unnamed_second_return']['shape'],[0])
            self.assertEqual(qli['configuration']['candidate_mode_L2'],3)
            self.assertEqual(qli['configuration']['candidate_mode_L20'],1)
            qli_source=next(s for s in result['sources'] if s['id']=='deepseek41-native-query-key-selection-20261109')
            self.assertEqual(qli_source['source_sha256'],m.sha((m.ROOT/'site/data/deepseek41-native-query-key-selection-20261109.json').read_bytes()))
            qli_asset=next(a for a in result['assets'] if a['file']=='supplements/deepseek41-native-query-key-selection-20261109.zip')
            self.assertEqual(qli_asset['source_sha256'],qli['candidate']['zip_sha256'])
            self.assertEqual(m.sha((out/qli_asset['file']).read_bytes()),qli['candidate']['zip_sha256'])
            self.assertIn('not native own full scores',qli_asset['scope'])
            self.assertIn('2Kall256candidate trivial',qli_asset['scope'])
            self.assertTrue(any('native producer-to-key/cache supplement' in x and 'query/score/top-k' in x for x in result['notes']))
            self.assertTrue(any('does not rewrite the earlier producer-key supplement scope' in x and 'native own full scores remain unavailable' in x for x in result['notes']))
            mode2=sources['deepseek41-retained-mode2-20261109']
            self.assertEqual(mode2['configuration']['candidate_mode'],2)
            self.assertEqual(mode2['configuration']['retained_query_layer'],20)
            self.assertEqual(mode2['numerics']['source_arrays_verified'],20)
            self.assertEqual(mode2['numerics']['actual_numeric_raw_pairs_verified'],21)
            self.assertEqual(mode2['numerics']['all_derived_score_coordinates'],131072)
            self.assertEqual([q['stable_reference_set_overlap'] for q in mode2['numerics']['selection_diagnostics']],[512,512,511,511])
            for key in ['original_crosslayer_chain','same_live_producer','attention','whole_model','TP','training','hardware_cause','native_own_scores_available','whole_task_completed']:
                self.assertFalse(mode2['scope'][key],key)
            self.assertIsNone(mode2['scope']['numerical_acceptance'])
            mode2_asset=next(a for a in result['assets'] if a['file']=='supplements/deepseek41-retained-mode2-20261109.zip')
            self.assertEqual(mode2_asset['source_sha256'],mode2['writing_delivery']['source_zip_sha256'])
            self.assertEqual(m.sha((out/mode2_asset['file']).read_bytes()),mode2['writing_delivery']['source_zip_sha256'])
            self.assertIn('Not the original crosslayer chain',mode2_asset['scope'])
            six=sources['deepseek41-six-forward-selectors-20261109']
            self.assertEqual(six['scope']['existing_forward_count'],6)
            self.assertEqual(six['scope']['selector_cases'],30)
            self.assertEqual(six['scope']['selected_query_cases'],120)
            self.assertEqual(six['scope']['full_derived_score_coordinates'],1740800)
            for key in ['independent_numeric_replay_in_this_ZIP','native_own_scores_available','original_default_unchunked32K_closed','full_history_queries_replayed','allTP_math_replayed','hardware_cause','whole_task_completed']:
                self.assertFalse(six['scope'][key],key)
            self.assertIsNone(six['scope']['numerical_acceptance'])
            self.assertTrue(six['scope']['original_single32768_forward_is_adapted_configuration'])
            self.assertEqual(len(six['records']),30)
            self.assertEqual(sum(len(r['selection']) for r in six['records']),120)
            self.assertEqual(sum(r['full_coordinates'] for r in six['records']),1740800)
            expected=[(2048,'all-visible-blocks',1792),(1024,'all-visible-blocks',1920),(2048,'all-visible-blocks',1792),(32768,'selective-2048-blocks',0),(16384,'all-visible-blocks',0),(32768,'selective-2048-blocks',0)]
            self.assertEqual([(c['key_rows'],c['candidate_branch'],c['negative_padding_per_query'][0]) for c in six['cohorts']],expected)
            self.assertTrue(all(c['negative_padding_per_query']==[e[2]]*4 for c,e in zip(six['cohorts'],expected)))
            six_asset=next(a for a in result['assets'] if a['file']=='supplements/deepseek41-six-forward-selectors-20261109.zip')
            self.assertEqual(six_asset['source_sha256'],'9c1f853fbd0e7352a0ea150e6f2561e4a1a412da2bf1fcb35d08f33cad10ea0d')
            self.assertEqual(m.sha((out/six_asset['file']).read_bytes()),six_asset['source_sha256'])
            self.assertIn('Byte/CSV verification only',six_asset['scope'])
            self.assertTrue(any('same-L20 mode2 supplement' in n and 'original L20-to-L24/28/32/36 crosslayer chain' in n for n in result['notes']))
            self.assertTrue(any('six-forward selector supplement' in n and 'not independent numerical replay' in n for n in result['notes']))
            with zipfile.ZipFile(out/http_asset['file']) as raw_rows:
                self.assertEqual(sum(n.startswith('arrays/') and n.endswith('.npy') for n in raw_rows.namelist()),128)
                self.assertIn('target-scalars.npy',raw_rows.namelist())
            history={x['id']:x for x in sources['model-evaluations']['evaluations']}
            self.assertIn('deepseek41-adapted-core275-20261109',history)
            self.assertEqual(history['deepseek41-original-vs-adapted-core-20261106']['comparisons'][0]['coverage'][0]['observed'],191)
            for p in out.rglob('*'):
                if p.is_file() and p.suffix!='.zip': self.assertIsNone(m.PRIVATE.search(p.read_text(encoding='utf-8-sig')),str(p))
            for report in (out/'reports').glob('*.md'):
                for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',report.read_text()):
                    if not re.match(r'[a-z]+:|#',link):
                        self.assertTrue((report.parent/link).resolve().is_file(),(report.name,link))
            with zipfile.ZipFile(out/'paper-writing-results.zip') as z:
                self.assertIsNone(z.testzip())
                self.assertIn('manifest.json',z.namelist())
                for stem in m.SUPPLEMENTS:
                    name='supplements/'+stem+'.zip'
                    original=(m.ROOT/'site/evidence'/(stem+'.zip')).read_bytes()
                    self.assertEqual(z.read(name),original)
            subprocess.run([sys.executable,'-I','-B',str(out/'verify.py')],check=True)
            (out/'values.csv').write_text('tampered')
            self.assertNotEqual(subprocess.run([sys.executable,'-I','-B',str(out/'verify.py')],capture_output=True).returncode,0)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for page rendering regression")
    def test_local_supplement_has_no_online_archive_download(self):
        script = "const fs=require('fs'),vm=require('vm'),assert=require('assert');\nconst nodes=new Map();class N{constructor(tag){this.tag=tag;this.children=[];this.value='';this.classList={add(){}};}append(...v){this.children.push(...v);}replaceChildren(...v){this.children=[...v];}}\nglobal.document={createElement:t=>new N(t),getElementById:id=>{if(!nodes.has(id))nodes.set(id,new N(id));return nodes.get(id);}};\nconst fixtureData={sources:[],numeric_fields:[],highlights:[],notes:[],assets:[{kind:'portable_supplement',file:'supplements/deepseek41-paired-output-head95-20261109.zip',scope:'Independent aggregate ZIP, not Markdown'},{kind:'report',file:'reports/native.md',text:'Retained native report'},{kind:'figure',file:'figures/plot.svg',svg:'<svg/>'}]};\nglobal.fetch=async()=>({ok:true,json:async()=>fixtureData});\nconst html=fs.readFileSync('site/paper.html','utf8');vm.runInThisContext(html.split('<script>')[1].split('</script>')[0]);\nsetImmediate(()=>{const reports=nodes.get('reports');assert.equal(reports.children.length,2);const zip=reports.children[0];const anchors=zip.children.filter(n=>n.tag==='a');assert.equal(anchors.length,0,'local archives must not have online download anchors');assert(zip.children.some(n=>n.textContent==='原始证据本地保留，不提供在线下载')); assert.equal(zip.children.filter(n=>n.tag==='button').length,0,'ZIP must not use Markdown preview/text Blob');assert.equal(reports.children[1].children.filter(n=>n.tag==='button').length,2);assert.equal(nodes.get('figures').children.length,1);console.log('PASS actual page async asset rendering: local-only archive, report and SVG');});\n"
        result=subprocess.run(["node","-e",script],cwd=m.ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("PASS actual page async asset rendering",result.stdout)
