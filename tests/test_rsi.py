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


if __name__=='__main__':
    unittest.main()
