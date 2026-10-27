"""Native subprocess tests for task continuation and evidence-gated experience reuse."""
import json
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from precision_rsi.loop import _prompt, _render, _usage, iterate, run_task
from precision_rsi.memory import evaluate, improvement, propose, read, select
from precision_rsi.evidence import collect, usage_totals
from precision_rsi.dataset import build as build_dataset
from precision_rsi import versions
from precision_rsi.improve import improve
from precision_rsi.benchmark import run as benchmark
from precision_rsi.harness import activity, price


class TaskLoopTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.workspace = self.root / 'session'
        self.task = self.root / 'task.json'
        (self.root / 'implementation.py').write_text('def objective(values): return sum(values)\n')
        (self.root / 'data.json').write_text('[1, 3]')
        (self.root / 'native.py').write_text('''import json,sys
from pathlib import Path
from implementation import objective
kind,out=sys.argv[1],Path(sys.argv[2])
if kind=='environment': result={'ready':True}
else: result={'value':objective(json.loads(Path('data.json').read_text()))}
out.write_text(json.dumps(result))
''')
        (self.root / 'verify.py').write_text('''import json,sys
from pathlib import Path
actual=json.loads(Path(sys.argv[1]).read_text())['value']
Path(sys.argv[2]).write_text(json.dumps({'correct':actual==2,'actual':actual}))
''')
        (self.root / 'agent.py').write_text('''import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text().split('\\n',1)[1])
action=sys.argv[2]
if action=='repair':
 Path('implementation.py').write_text('def objective(values): return sum(values)/len(values)\\n')
 Path(p['proposal']).write_text(json.dumps({'kind':'context','content':'Use the objective reduction defined by the task.','tags':['mean']}))
elif action=='forge':
 Path('verify.py').write_text('print("changed verifier")')
elif action=='bad-proposal':
 Path('implementation.py').write_text('def objective(values): return sum(values)/len(values)\\n')
 Path(p['proposal']).write_text('invalid json')
elif action=='repair-then-fail':
 Path('implementation.py').write_text('def objective(values): return sum(values)/len(values)\\n')
 raise SystemExit(7)
elif action=='repair-then-timeout':
 Path('implementation.py').write_text('def objective(values): return sum(values)/len(values)\\n')
 import time
 time.sleep(10)
print(json.dumps({'usage':{'input_tokens':12,'output_tokens':4}}))
''')
        plan = dict(schema_version=1, id='mean', budget_seconds=30, stages=[
            dict(id='environment', skill='framework-adapt', timeout=5, inputs=['native.py'],
                 argv=[sys.executable, '-B', 'native.py', 'environment', '{run}/ready.json'],
                 outputs={'ready':'ready.json'}, checks=[dict(output='ready',pointer='/ready',equals=True)]),
            dict(id='reproduce', skill='precision-compare', timeout=5, depends_on=['environment'],
                 inputs=['native.py','implementation.py','data.json'],
                 argv=[sys.executable,'-B','native.py','reproduce','{run}/capture.json'],
                 outputs={'capture':'capture.json'}),
            dict(id='compare', skill='precision-compare', timeout=5, depends_on=['reproduce'],
                 inputs=['verify.py'],
                 argv=[sys.executable,'-B','verify.py','{artifact:reproduce:capture}','{run}/metrics.json'],
                 outputs={'metrics':'metrics.json'},checks=[dict(output='metrics',pointer='/correct',equals=True)],
                 on_check_failure='precision-repair')])
        (self.root/'native-plan.json').write_text(json.dumps(plan))
        self.cfg = dict(schema_version=1,goal='Repair the mean objective and verify it.',plan='native-plan.json',
                budget=dict(seconds=30,agent_calls=3),
                agent=dict(argv=[sys.executable,'-B','agent.py','{prompt_file}','repair'],timeout=5),
                protected=['native.py','verify.py','data.json'],memory=dict(directory='memory',tags=['mean']))
        self.save()

    def save(self):
        self.task.write_text(json.dumps(self.cfg))

    def test_repairs_then_reuses_original_checks_without_another_agent(self):
        first = run_task(self.task,self.workspace)
        self.assertEqual(first['status'],'completed')
        self.assertEqual(first['agent_calls'],1)
        self.assertEqual(read(self.root/'memory'/(first['candidate']+'.json'))['status'],'candidate')
        self.assertEqual(select(self.root/'memory',['mean']),[])
        native = read(self.workspace/'native/state.json')['history']
        metrics = [read(a['outputs']['metrics']['path']) for a in native
                   if a['stage']=='compare' and a.get('outputs')]
        self.assertEqual([m['correct'] for m in metrics],[False,True])
        self.assertEqual(sum(a['stage']=='environment' for a in native),1)
        with patch('precision_rsi.loop.record_run',side_effect=AssertionError('agent rerun')):
            replay = run_task(self.task,self.workspace)
        self.assertEqual(replay['status'],'completed')
        self.assertEqual(replay['agent_calls'],1)
        self.assertTrue(all(s['reused'] for s in replay['workflow']['stages']))

    def test_zero_agent_budget_leaves_action_and_original_failure(self):
        self.cfg['budget']['agent_calls']=0;self.save()
        result=run_task(self.task,self.workspace)
        self.assertEqual(result['reason'],'agent call budget exhausted')
        self.assertEqual(result['agent_calls'],0)
        self.assertEqual(result['workflow']['next_action']['skill'],'precision-repair')

    def test_no_change_stops_after_one_unsuccessful_action(self):
        self.cfg['agent']['argv'][-1]='noop';self.save()
        result=run_task(self.task,self.workspace)
        self.assertEqual(result['reason'],'agent made no verified progress')
        self.assertEqual(result['agent_calls'],1)

    def test_partial_agent_failure_still_verifies_saved_patch(self):
        self.cfg['agent']['argv'][-1]='repair-then-fail';self.save()
        result=run_task(self.task,self.workspace)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['agent_calls'],1)
        self.assertTrue(read(result['workflow']['stages'][-1]['outputs']['metrics']['path'])['correct'])

    def test_timeout_still_verifies_saved_patch_with_remaining_budget(self):
        self.cfg['agent']['argv'][-1]='repair-then-timeout'
        self.cfg['agent']['timeout']=1;self.save()
        result=run_task(self.task,self.workspace)
        self.assertEqual(result['status'],'completed')
        history=read(self.workspace/'loop.json')['history']
        self.assertEqual([a['status'] for a in history if a['kind']=='agent'],['timed_out'])

    def test_agent_cannot_complete_by_rewriting_verification(self):
        self.cfg['agent']['argv'][-1]='forge';self.save()
        result=run_task(self.task,self.workspace)
        self.assertEqual(result['status'],'needs_action')
        self.assertEqual(result['reason'],'protected task or evaluator changed')
        self.assertIsNone(result['candidate'])

    def test_bad_optional_experience_does_not_undo_completed_task(self):
        self.cfg['agent']['argv'][-1]='bad-proposal';self.save()
        result=run_task(self.task,self.workspace)
        self.assertEqual(result['status'],'completed')
        self.assertIn('candidate_error',result)

    def test_budget_extension_keeps_native_history(self):
        self.cfg['budget']['agent_calls']=0;self.save()
        before=run_task(self.task,self.workspace)
        self.cfg['budget']['agent_calls']=1;self.save()
        after=run_task(self.task,self.workspace)
        self.assertEqual(after['status'],'completed')
        self.assertTrue(after['workflow']['stages'][0]['reused'])
        self.assertEqual(before['agent_calls'],0)

    def test_open_search_is_not_reported_as_implemented(self):
        self.cfg['mode']='open';self.save()
        with self.assertRaisesRegex(ValueError,'operator-optimize'):
            run_task(self.task,self.workspace)

    def test_harness_usage_counts_step_events_without_double_counting_summary(self):
        path=self.root/'usage.jsonl'
        event=dict(type='status',phase='step_end',usage=dict(inputTokens=12,outputTokens=4,totalTokens=16))
        path.write_text('ordinary log\n'+json.dumps(event)+'\n'+json.dumps(dict(event,type='result'))+'\n')
        self.assertEqual(_usage(path,'dsh-jsonl'),dict(input_tokens=12,output_tokens=4,total_tokens=16,
                                                    steps=1,cached_input_tokens=None))
        self.assertIsNone(_usage(path,None))

    def test_delegation_supplies_proposal_interface_and_keeps_failure_evidence(self):
        action=dict(stage='compare',skill='precision-repair',reason='check failed',
                    execution=dict(status='completed',stderr='warning'),checks=['verbose criteria'])
        prompt=_prompt(self.cfg,self.root/'native-plan.json',action,self.workspace,[])
        self.assertIn('"kind":"context"',prompt)
        self.assertIn('absolute paths',prompt)
        payload=json.loads(prompt.split('\n',1)[1])
        self.assertEqual(payload['action']['reason'],'check failed')
        self.assertNotIn('execution',payload['action'])
        action['execution']['status']='failed'
        self.assertEqual(json.loads(_prompt(self.cfg,self.task,action,self.workspace,[])
                                   .split('\n',1)[1])['action']['execution']['status'],'failed')

    def test_evaluator_template_preserves_quoted_paths_as_json_values(self):
        result=_render({'argv':['--candidate','{candidate_file}'],'id':'{candidate_id}'},
                       {'{candidate_file}':'/work/a "quoted" path/c.json','{candidate_id}':'abc'})
        self.assertEqual(json.loads(json.dumps(result))['argv'][1],'/work/a "quoted" path/c.json')
        self.assertEqual(result['id'],'abc')

    def test_evidence_includes_incomplete_tasks_without_exposing_raw_logs(self):
        self.cfg['budget']['agent_calls']=0;self.cfg['family']='objective-reduction';self.save()
        run_task(self.task,self.workspace)
        index=collect([self.workspace])
        item=index['tasks'][0]
        self.assertEqual(item['status'],'needs_action')
        self.assertEqual(item['family'],'objective-reduction')
        self.assertEqual(item['agent_calls'],0)
        self.assertTrue(item['references'])
        observed=[v['values'] for v in item['observations'] if v['output']=='metrics']
        self.assertEqual(observed[0]['correct'],False)
        before=index['id']
        run_task(self.task,self.workspace)
        self.assertEqual(collect([self.workspace])['id'],before)

    def test_activity_summarizes_actions_without_exposing_commands_or_contents(self):
        path=self.root/'actions.jsonl'
        events=[dict(type='tool_call',callId='read1',tool='read',input=dict(file_path='/work/score_metrics.py')),
                dict(type='tool_call',callId='cmd1',tool='bash',input=dict(command='TOKEN=private-value python compare.py; ls *.py')),
                dict(type='tool_call',callId='cmd2',tool='bash',input=dict(command='TOKEN=private-value python compare.py; ls *.py'))]
        path.write_text('\n'.join(json.dumps(e) for e in events+[events[0]])+'\nmalformed\n')
        result=activity(path,'dsh-jsonl')
        self.assertEqual(result['tool_calls'],3)
        self.assertEqual(result['read_files'],{'score_metrics.py':1})
        self.assertEqual(result['python_mentions'],{'compare.py':2})
        self.assertEqual(result['repeated_commands'],1)
        self.assertNotIn('private-value',json.dumps(result))

    def test_evidence_retains_bounded_agent_activity(self):
        self.cfg['agent']['usage_format']='dsh-jsonl';self.save();run_task(self.task,self.workspace)
        state=read(self.workspace/'loop.json')
        attempt=next(a for a in state['history'] if a['kind']=='agent')
        with Path(attempt['raw_output']).open('a') as stream:
            stream.write(json.dumps(dict(type='tool_call',tool='read',input=dict(file_path='/work/plan.json')))+'\n')
        summary=collect([self.workspace])['tasks'][0]['agent_activity']
        self.assertEqual(summary[0]['read_files'],{'plan.json':1})

    def dataset_source(self):
        self.cfg['family']='objective-reduction';self.save()
        run_task(self.task,self.workspace)
        index=collect([self.workspace])
        item=dict(source=index['tasks'][0]['id'],messages=[dict(role='user',content='Repair mean loss.'),
                                                        dict(role='assistant',content='Use the mean reduction.')])
        filename=self.root/'examples.jsonl'
        filename.write_text(json.dumps(item)+'\n'+json.dumps(item)+'\n')
        return index,item,filename

    def test_dataset_deduplicates_and_uses_declared_family_split(self):
        index,item,filename=self.dataset_source()
        result=build_dataset([filename],index,self.root/'data',splits={'objective-reduction':'train'})
        self.assertEqual(result['counts'],dict(train=1,validation=0,test=0))
        self.assertEqual(result['duplicates'],1)
        record=json.loads(Path(result['artifacts'][0]).read_text())
        self.assertEqual(record['outcome'],'completed')
        self.assertEqual(record['label_scope'],'task')
        replay=build_dataset([filename],index,self.root/'data',splits={'objective-reduction':'train'})
        self.assertEqual(result['id'],replay['id'])

    def test_dataset_rejects_holdout_families_and_forged_outcomes(self):
        index,item,filename=self.dataset_source()
        with self.assertRaisesRegex(ValueError,'held-out'):
            build_dataset([filename],index,self.root/'data',['objective-reduction'])
        item['outcome']='needs_action';filename.write_text(json.dumps(item)+'\n')
        with self.assertRaisesRegex(ValueError,'outcome'):
            build_dataset([filename],index,self.root/'data')

    def test_dataset_rejects_modified_native_evidence(self):
        index,item,filename=self.dataset_source()
        evidence=next(iter(index['tasks'][0]['references']))
        Path(evidence).write_text('changed capture')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            build_dataset([filename],index,self.root/'data')

    def test_unknown_usage_is_not_counted_as_zero(self):
        result=usage_totals([dict(usage=dict(input_tokens=10,output_tokens=2,total_tokens=12,
                                           cached_input_tokens=None)),dict(usage=None)])
        self.assertIsNone(result['total_tokens'])
        self.assertIsNone(result['cached_input_tokens'])

    def test_task_pins_candidate_version_and_procedure_route(self):
        skill=self.root/'short-skill.md';skill.write_text('Use the existing objective interface.')
        policy=self.root/'policy.json';policy.write_text(json.dumps(dict(
            instruction='Use the objective interface.',skill_paths={'precision-repair':str(skill)})))
        store=self.root/'memory'
        identity=propose(store,dict(kind='procedure',content='Short repair procedure.',tags=['mean'],
            artifacts=[str(policy),str(skill)],entrypoint=str(policy)),{})
        version=versions.prepare(store,[identity])
        first=run_task(self.task,self.workspace,system_version=version['id'])
        self.assertEqual(first['status'],'completed')
        state=read(self.workspace/'loop.json')
        call=next(a for a in state['history'] if a['kind']=='agent')
        prompt=Path(call['run'])/'prompt.txt'
        self.assertEqual(json.loads(prompt.read_text().split('\n',1)[1])['skill'],str(skill))
        with patch('precision_rsi.loop.record_run',side_effect=AssertionError('replay agent')):
            self.assertEqual(run_task(self.task,self.workspace)['system_version'],version['id'])
        with self.assertRaisesRegex(ValueError,'version changed'):
            run_task(self.task,self.workspace,system_version='empty')

    def test_stream_admits_experience_before_the_next_task(self):
        other = tempfile.TemporaryDirectory();self.addCleanup(other.cleanup)
        peer = Path(other.name)/'next'
        shutil.copytree(self.root,peer)
        self.cfg['memory']['directory']=str(self.root/'memory');self.save()
        (peer/'task.json').write_text(json.dumps(self.cfg))
        (self.root/'evaluate.py').write_text('''import json,sys
from pathlib import Path
Path(sys.argv[2]).write_text(json.dumps({'candidate_id':sys.argv[1],'mode':'closed','scope':['independent-evaluation-fixture'],
 'baseline':{'correct':True,'seconds':10},'candidate':{'correct':True,'seconds':8}}))
''')
        (self.root/'eval-template.json').write_text(json.dumps(dict(schema_version=1,id='evaluate',budget_seconds=10,
            workload=dict(purpose='held_out_evaluation',candidate_id='{candidate_id}'),stages=[
                dict(id='evaluate',skill='precision-compare',timeout=5,inputs=['evaluate.py'],
                     argv=[sys.executable,'-B','evaluate.py','{candidate_id}','{run}/evaluation.json'],
                     outputs={'evaluation':'evaluation.json'})])))
        series=self.root/'series.json'
        series.write_text(json.dumps(dict(schema_version=1,tasks=[str(self.task),str(peer/'task.json')],
            budget_seconds=30,evaluation_plan='eval-template.json')))
        result=iterate(series,self.root/'stream')
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['tasks'][0]['evaluation']['status'],'active')
        second=read(self.root/'stream/tasks/1/loop.json')
        calls=[a for a in second['history'] if a['kind']=='agent']
        self.assertEqual(calls[0]['memory_ids'],[result['tasks'][0]['result']['candidate']])
        with patch('precision_rsi.loop.record_run',side_effect=AssertionError('unexpected agent replay')):
            self.assertEqual(iterate(series,self.root/'stream')['status'],'completed')


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.store=self.root/'memory'
        self.tool=self.root/'tool.py';self.tool.write_text('print("tool")')
        self.identity=propose(self.store,dict(kind='tool',content='Run tool.py for the requested statistic.',
            tags=['torch'],artifacts=[str(self.tool)]),dict(task='training-task'))
        self.measurement=dict(candidate_id=self.identity,mode='closed',scope=['held-out-task'],
                baseline=dict(correct=True,seconds=10,cost=2),candidate=dict(correct=True,seconds=8,cost=1))

    def evaluate(self):
        (self.root/'measurement.json').write_text(json.dumps(self.measurement))
        (self.root/'evaluator.py').write_text('import shutil,sys\nshutil.copyfile("measurement.json",sys.argv[1])\n')
        self.plan=self.root/'evaluation.json'
        self.plan.write_text(json.dumps(dict(schema_version=1,id='evaluation',budget_seconds=10,
            workload=dict(purpose='held_out_evaluation',candidate_id=self.identity),stages=[
                dict(id='measure',skill='precision-compare',timeout=5,inputs=['measurement.json','evaluator.py'],
                argv=[sys.executable,'-B','evaluator.py','{run}/evaluation.json'],outputs={'evaluation':'evaluation.json'})])))
        return evaluate(self.store,self.identity,self.plan,self.root/'evaluation-workspace')

    def test_only_evaluated_unchanged_applicable_tools_enter_memory(self):
        self.assertEqual(select(self.store,['torch']),[])
        result=self.evaluate()
        self.assertEqual(result['status'],'active')
        self.assertEqual(len(select(self.store,['torch'])),1)
        self.assertEqual(select(self.store,['other']),[])
        self.assertEqual(select(self.store,['torch'],10),[])
        self.tool.write_text('print("modified")')
        self.assertEqual(select(self.store,['torch']),[])

    def test_correctness_failure_rejects_and_removes_admission(self):
        self.evaluate()
        self.measurement['candidate']['correct']=False
        self.assertEqual(self.evaluate()['status'],'candidate')
        self.assertEqual(select(self.store,['torch']),[])

    def test_slower_or_more_expensive_candidate_is_not_admitted(self):
        self.measurement['candidate']['cost']=3
        self.assertEqual(self.evaluate()['status'],'candidate')

    def test_measurements_cannot_reference_another_candidate(self):
        self.measurement['candidate_id']='0'*64
        with self.assertRaisesRegex(ValueError,'another candidate'):
            self.evaluate()

    def test_evaluation_binding_and_scope_are_required(self):
        self.evaluate()
        plan=read(self.plan);plan['workload']['purpose']='training';self.plan.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'held-out'):
            evaluate(self.store,self.identity,self.plan,self.root/'other')
        self.measurement['scope']=[]
        with self.assertRaisesRegex(ValueError,'scope'):
            improvement(self.measurement)

    def test_open_search_compares_quality_under_equal_budget(self):
        result=dict(mode='open',scope=['operator'],baseline=dict(correct=True,seconds=10,budget=30,quality=2),
                    candidate=dict(correct=True,seconds=12,budget=30,quality=3))
        self.assertTrue(improvement(result))
        result['candidate']['budget']=60
        self.assertFalse(improvement(result))

    def test_unequal_or_unknown_costs_are_not_assumed_free(self):
        self.measurement['candidate'].pop('cost')
        with self.assertRaisesRegex(ValueError,'both costs'):
            improvement(self.measurement)

    def test_admission_accounts_for_token_regressions(self):
        self.measurement['baseline']['total_tokens']=100
        self.measurement['candidate']['total_tokens']=120
        self.assertFalse(improvement(self.measurement))

    def test_pricing_requires_cache_attribution_when_rates_differ(self):
        usage=dict(input_tokens=100,output_tokens=10,cached_input_tokens=None)
        rates=dict(input_per_million=2,output_per_million=3,cached_input_per_million=1)
        self.assertIsNone(price(usage,rates))
        usage['cached_input_tokens']=50
        self.assertAlmostEqual(price(usage,rates),0.00018)
        usage['cached_input_tokens']=None;rates['cached_input_per_million']=2
        self.assertAlmostEqual(price(usage,rates),0.00023)

    def version_evaluation(self, version, correct=True):
        measurement=dict(version_id=version['id'],baseline_version=version['base_version'],
            mode='closed',scope=['independent-fixture'],baseline=dict(correct=True,seconds=10),
            candidate=dict(correct=correct,seconds=8))
        (self.root/'version-measurement.json').write_text(json.dumps(measurement))
        (self.root/'copy.py').write_text('import shutil,sys\nshutil.copyfile("version-measurement.json",sys.argv[1])\n')
        plan=self.root/'version-eval.json'
        plan.write_text(json.dumps(dict(schema_version=1,id='version-eval',budget_seconds=10,
            workload=dict(purpose='held_out_evaluation',version_id=version['id'],baseline_version=version['base_version']),
            stages=[dict(id='evaluate',skill='precision-compare',timeout=5,inputs=['copy.py','version-measurement.json'],
                argv=[sys.executable,'-B','copy.py','{run}/evaluation.json'],outputs={'evaluation':'evaluation.json'})])))
        return versions.evaluate(self.store,version['id'],plan,self.root/'version-evaluation')

    def test_bundle_requires_bound_evaluation_before_activation(self):
        version=versions.prepare(self.store,[self.identity])
        self.assertEqual(select(self.store,['torch'],version=version['id'])[0]['id'],self.identity)
        self.assertEqual(versions.current(self.store),'empty')
        with self.assertRaises(OSError):
            versions.activate(self.store,version['id'])
        self.assertEqual(self.version_evaluation(version)['status'],'active')
        self.assertEqual(versions.current(self.store),version['id'])
        self.assertEqual(select(self.store,['torch'])[0]['id'],self.identity)
        self.assertEqual(select(self.store,['torch'],version='empty'),[])

    def test_failed_update_preserves_current_version_and_can_restore_old_version(self):
        first=versions.prepare(self.store,[self.identity]);self.version_evaluation(first)
        other=propose(self.store,dict(kind='context',content='A bounded instruction.',tags=['torch']),{})
        second=versions.prepare(self.store,[other])
        self.assertEqual(self.version_evaluation(second,False)['status'],'candidate')
        self.assertEqual(versions.current(self.store),first['id'])
        self.assertEqual(self.version_evaluation(second)['status'],'active')
        versions.activate(self.store,first['id'])
        self.assertEqual(versions.current(self.store),first['id'])

    def test_pinned_version_rejects_changed_artifact_or_candidate(self):
        version=versions.prepare(self.store,[self.identity])
        self.tool.write_text('changed tool')
        with self.assertRaisesRegex(ValueError,'changed'):
            select(self.store,['torch'],version=version['id'])

    def test_empty_baseline_can_be_restored_without_a_synthetic_evaluation(self):
        self.assertEqual(versions.activate(self.store,'empty')['id'],'empty')
        first=versions.prepare(self.store,[self.identity]);self.version_evaluation(first)
        result=versions.activate(self.store,'empty',expected_current=first['id'])
        self.assertEqual(result['previous'],first['id'])
        self.assertEqual(versions.current(self.store),'empty')
        self.assertEqual(select(self.store,['torch']),[])
        versions.activate(self.store,first['id'])
        with self.assertRaisesRegex(ValueError,'active base advanced'):
            versions.activate(self.store,'empty',expected_current='empty')

    def test_procedure_references_only_versioned_skill_files(self):
        skill=self.root/'SKILL.md';skill.write_text('Use the supplied tool interface.')
        policy=self.root/'policy.json';policy.write_text(json.dumps(dict(instruction='Reuse the capture.',
                                                        skill_paths={'precision-repair':str(skill)})))
        proposal=dict(kind='procedure',content='Use the short repair route.',tags=['torch'],
                      entrypoint=str(policy),artifacts=[str(policy)])
        with self.assertRaisesRegex(ValueError,'declared skill'):
            propose(self.store,proposal,{})
        proposal['artifacts'].append(str(skill))
        identity=propose(self.store,proposal,{})
        version=versions.prepare(self.store,[identity])
        memory=select(self.store,['torch'],version=version['id'])
        self.assertEqual(memory[0]['procedure']['skill_paths']['precision-repair'],str(skill))


class ImprovementTests(unittest.TestCase):
    setUp = TaskLoopTests.setUp
    save = TaskLoopTests.save

    def setup_improvement(self, complete=True, mode='updates'):
        self.cfg['family']='source-objective'
        if not complete:
            self.cfg['budget']['agent_calls']=0
        self.save();run_task(self.task,self.workspace)
        self.improver=self.root/'improver.py'
        self.improver.write_text('''import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text().split('\\n',1)[1])
folder=Path(p['artifacts']);folder.mkdir(exist_ok=True)
tool=folder/'tool.py';tool.write_text('print("reusable")')
policy=folder/'policy.json';policy.write_text(json.dumps({'instruction':'Reuse the existing reduction diagnostic.'}))
data=folder/'examples.jsonl';data.write_text(json.dumps({'source':p['evidence'][0]['id'],
 'messages':[{'role':'user','content':'Diagnose the reduction.'},{'role':'assistant','content':'Inspect its normalization.'}]})+'\\n')
updates=[{'kind':'context','content':'Inspect reduction semantics.','tags':['mean'],'artifacts':[]},
 {'kind':'tool','content':'Run tool.py.','tags':['mean'],'artifacts':[str(tool)]},
 {'kind':'procedure','content':'Reuse diagnostics.','tags':['mean'],'entrypoint':str(policy),'artifacts':[str(policy)]},
 {'kind':'data','content':'Outcome-labeled task example.','tags':['mean'],'artifacts':[str(data)]}]
Path(p['output']).write_text(json.dumps({'updates':[] if sys.argv[2]=='none' else updates}))
print(json.dumps({'type':'status','phase':'step_end','usage':{'inputTokens':10,'outputTokens':2,'totalTokens':12}}))
''')
        self.improve_cfg=self.root/'improve.json'
        cfg=dict(schema_version=1,store='memory',evidence=[str(self.workspace)],budget_seconds=20,
            agent=dict(argv=[sys.executable,'-B',str(self.improver),'{prompt_file}',mode],timeout=5,
                       usage_format='dsh-jsonl'),held_out_families=['unseen-objective'])
        self.improve_cfg.write_text(json.dumps(cfg))

    def test_independent_improver_builds_four_outputs_and_reuses_completed_call(self):
        self.setup_improvement()
        session=self.root/'improvement-session'
        first=improve(self.improve_cfg,session)
        self.assertEqual(first['status'],'completed')
        self.assertEqual(len(first['candidates']),4)
        self.assertEqual(first['usage']['total_tokens'],12)
        self.assertEqual(first['datasets'][0]['status'],'ready')
        self.assertEqual(versions.current(self.root/'memory'),'empty')
        candidates=[read(self.root/'memory'/(i+'.json')) for i in first['candidates']]
        self.assertEqual(next(v for v in candidates if v['kind']=='data')['status'],'ready')
        second=improve(self.improve_cfg,session)
        self.assertEqual(first['version'],second['version'])
        self.assertTrue(second['workflow']['stages'][0]['reused'])

    def test_failed_task_can_produce_outcome_labeled_data(self):
        self.setup_improvement(complete=False)
        result=improve(self.improve_cfg,self.root/'improvement-session')
        dataset=result['datasets'][0]
        rows=[json.loads(line) for p in dataset['artifacts'] for line in Path(p).read_text().splitlines()]
        self.assertEqual(rows[0]['outcome'],'needs_action')

    def test_no_update_is_a_successful_cached_result(self):
        self.setup_improvement(mode='none')
        result=improve(self.improve_cfg,self.root/'improvement-session')
        self.assertTrue(result['no_update'])
        self.assertIsNone(result['version'])
        self.assertTrue(improve(self.improve_cfg,self.root/'improvement-session')['workflow']['stages'][0]['reused'])

    def test_benchmark_runs_independent_arms_and_retains_actual_measurements(self):
        self.cfg['family']='unseen-objective';self.save()
        tasks={}
        for arm in ('baseline','candidate'):
            folder=self.root/arm;folder.mkdir()
            for name in ('task.json','native-plan.json','agent.py','implementation.py','data.json','verify.py','native.py'):
                shutil.copyfile(self.root/name,folder/name)
            tasks[arm]=str(folder/'task.json')
        store=self.root/'memory'
        identity=propose(store,dict(kind='context',content='Inspect reduction semantics.',tags=['mean']),
                         dict(families=['source-objective']))
        version=versions.prepare(store,[identity])
        suite=self.root/'suite.json'
        suite.write_text(json.dumps(dict(schema_version=1,store=str(store),cases=[dict(
            id='independent-mean',family='unseen-objective',**tasks)])))
        first=benchmark(suite,self.root/'bench',version['id'],'empty',30)
        self.assertTrue(first['baseline']['correct']);self.assertTrue(first['candidate']['correct'])
        self.assertGreater(first['baseline']['seconds'],0)
        with patch('precision_rsi.loop.record_run',side_effect=AssertionError('benchmark replay called agent')):
            self.assertEqual(benchmark(suite,self.root/'bench',version['id'],'empty',30),first)
        fresh=self.root/'fresh-candidate';fresh.mkdir()
        for name in ('task.json','native-plan.json','agent.py','implementation.py','data.json','verify.py','native.py'):
            shutil.copyfile(self.root/name,fresh/name)
        spec=read(suite)
        spec['cases'][0].pop('baseline')
        spec['cases'][0]['baseline_receipt']=str(self.root/'bench/0/baseline/result.json')
        spec['cases'][0]['candidate']=str(fresh/'task.json')
        suite2=self.root/'reuse-suite.json';suite2.write_text(json.dumps(spec))
        reused=benchmark(suite2,self.root/'reuse-bench',version['id'],'empty',30)
        self.assertEqual(reused['baseline'],first['baseline'])
        state=read(self.root/'reuse-bench/benchmark.json')
        self.assertEqual(state['arms']['0:baseline']['evidence']['state'],
                         read(self.root/'bench/0/baseline/result.json')['evidence']['state'])

    def test_benchmark_rejects_shared_workload_and_source_family(self):
        self.cfg['family']='source-objective';self.save()
        store=self.root/'memory'
        identity=propose(store,dict(kind='context',content='Inspect reduction semantics.',tags=['mean']),
                         dict(families=['source-objective']))
        version=versions.prepare(store,[identity])
        suite=self.root/'suite.json'
        spec=dict(schema_version=1,store=str(store),cases=[dict(id='same',family='source-objective',
            baseline=str(self.task),candidate=str(self.task))])
        suite.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError,'overlaps'):
            benchmark(suite,self.root/'bench',version['id'],'empty',30)
        spec['cases'][0]['family']='unseen';suite.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError,'independent workload'):
            benchmark(suite,self.root/'bench',version['id'],'empty',30)

    def test_benchmark_allows_agents_to_create_missing_implementations(self):
        (self.root/'implementation.py').unlink()
        tasks={}
        for arm in ('baseline','candidate'):
            folder=self.root/arm;folder.mkdir()
            for name in ('task.json','native-plan.json','agent.py','data.json','verify.py','native.py'):
                shutil.copyfile(self.root/name,folder/name)
            tasks[arm]=str(folder/'task.json')
        store=self.root/'memory'
        identity=propose(store,dict(kind='context',content='Use mean reduction.',tags=['mean']),
                         dict(families=['source-objective']))
        version=versions.prepare(store,[identity])
        suite=self.root/'missing-suite.json'
        suite.write_text(json.dumps(dict(schema_version=1,store=str(store),cases=[dict(
            id='create-implementation',family='unseen-objective',**tasks)])))
        result=benchmark(suite,self.root/'missing-bench',version['id'],'empty',30)
        for arm in ('baseline','candidate'):
            self.assertTrue(result[arm]['correct'])
            self.assertTrue((self.root/arm/'implementation.py').is_file())
        state=read(self.root/'missing-bench/benchmark.json')
        self.assertEqual(state['conditions']['0']['sources'][1][1],{'kind':'missing'})
        with patch('precision_rsi.loop.record_run',side_effect=AssertionError('agent rerun')):
            self.assertEqual(benchmark(suite,self.root/'missing-bench',version['id'],'empty',30),result)

    def test_benchmark_rejects_unselected_updates_before_calling_agents(self):
        tasks={}
        for arm in ('baseline','candidate'):
            folder=self.root/arm;folder.mkdir()
            for name in ('task.json','native-plan.json','agent.py','implementation.py','data.json','verify.py','native.py'):
                shutil.copyfile(self.root/name,folder/name)
            tasks[arm]=str(folder/'task.json')
        store=self.root/'memory'
        identity=propose(store,dict(kind='context',content='An unrelated instruction.',tags=['other-schema']),
                         dict(families=['source-objective']))
        version=versions.prepare(store,[identity])
        suite=self.root/'unrelated-suite.json'
        suite.write_text(json.dumps(dict(schema_version=1,store=str(store),cases=[dict(
            id='unrelated-tool',family='unseen-objective',**tasks)])))
        with patch('precision_rsi.loop.record_run',side_effect=AssertionError('agent should not run')):
            with self.assertRaisesRegex(ValueError,'no applicable runtime difference'):
                benchmark(suite,self.root/'unrelated-bench',version['id'],'empty',30)
        self.assertFalse((store/'versions/current.json').exists())

    def test_stream_publishes_bundle_for_the_next_task_without_replaying_improvement(self):
        self.setup_improvement()
        cfg=read(self.improve_cfg);cfg['evidence']=[];cfg['trigger']=dict(min_new_tasks=1)
        cfg['evaluation_plan']='bundle-eval.json'
        self.improve_cfg.write_text(json.dumps(cfg))
        (self.root/'implementation.py').write_text('def objective(values): return sum(values)\n')
        self.cfg['memory']['directory']=str(self.root/'memory');self.save()
        peer=self.root/'peer';peer.mkdir()
        for name in ('task.json','native-plan.json','agent.py','implementation.py','data.json','verify.py','native.py'):
            shutil.copyfile(self.root/name,peer/name)
        (self.root/'eval.py').write_text('''import json,sys
from pathlib import Path
Path(sys.argv[3]).write_text(json.dumps({'version_id':sys.argv[1],'baseline_version':sys.argv[2],
 'mode':'closed','scope':['independent-admission-fixture'],'families':['unseen-objective'],
 'baseline':{'correct':False,'seconds':10},'candidate':{'correct':True,'seconds':8}}))
''')
        (self.root/'bundle-eval.json').write_text(json.dumps(dict(schema_version=1,id='bundle-eval',budget_seconds=10,
            workload=dict(purpose='held_out_evaluation',version_id='{version_id}',baseline_version='{baseline_version}'),
            stages=[dict(id='evaluate',skill='precision-compare',timeout=5,inputs=['eval.py'],
                argv=[sys.executable,'-B','eval.py','{version_id}','{baseline_version}','{run}/evaluation.json'],
                outputs={'evaluation':'evaluation.json'})])))
        series=self.root/'series.json'
        series.write_text(json.dumps(dict(schema_version=1,tasks=[str(self.task),str(peer/'task.json')],
                            budget_seconds=40,improvement=str(self.improve_cfg))))
        result=iterate(series,self.root/'stream')
        self.assertEqual(result['status'],'completed')
        first=result['tasks'][0]['improvement']
        self.assertEqual(first['evaluation']['status'],'active')
        second=read(self.root/'stream/tasks/1/loop.json')
        self.assertEqual(second['system_version'],first['version'])
        self.assertTrue(next(a for a in second['history'] if a['kind']=='agent')['memory_ids'])
        state=read(self.root/'stream/series.json')
        with patch('precision_rsi.loop.record_run',side_effect=AssertionError('replay called agent')):
            replay=iterate(series,self.root/'stream')
        self.assertEqual(replay['status'],'completed')
        self.assertEqual(len(read(self.root/'stream/series.json')['improvements']),len(state['improvements']))


if __name__=='__main__':
    unittest.main()
