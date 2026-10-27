"""External controller feedback, stopping and replay regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from precision_rsi.evolve import _feedback, run
from precision_rsi.memory import propose, read


class EvolutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root/'config.json'
        self.cfg = dict(schema_version=1,store='memory',budget_seconds=30,round_seconds=5,
            max_rounds=4,patience=3,evidence=[])
        self.save()

    def save(self):
        self.config.write_text(json.dumps(self.cfg))

    def test_rejection_reaches_next_round_and_replay_calls_nothing(self):
        received = []
        def improve(config, workspace, **kwargs):
            received.append(read(config))
            if len(received)==1:
                return dict(status='needs_action',errors=[dict(reason='not applicable')],candidates=[])
            return dict(status='completed',no_update=True,candidates=[])
        with patch('precision_rsi.evolve.improve',side_effect=improve):
            result=run(self.config,self.root/'session')
        self.assertEqual(result['rounds'],2)
        self.assertIn('not applicable',json.dumps(received[1]['feedback']))
        self.assertEqual(result['reason'],'no useful update')
        with patch('precision_rsi.evolve.improve',side_effect=AssertionError('replayed model')):
            self.assertEqual(run(self.config,self.root/'session'),result)

    def test_repeated_failure_stops(self):
        with patch('precision_rsi.evolve.improve',return_value=dict(status='completed',candidates=[])) as agent:
            result=run(self.config,self.root/'session')
        self.assertEqual(agent.call_count,2)
        self.assertEqual(result['reason'],'repeated failure')

    def test_improver_launch_failure_reaches_next_round(self):
        calls=[]
        failure=dict(status='needs_action',next_action=dict(stage='improve',
            reason='native command failed',execution=dict(status='launch_failed')))
        def improve(config,workspace,**kwargs):
            calls.append(read(config))
            if len(calls)==1:
                return dict(status='needs_action',candidates=[],workflow=failure)
            return dict(status='completed',candidates=[],no_update=True)
        with patch('precision_rsi.evolve.improve',side_effect=improve):
            result=run(self.config,self.root/'launch-failure')
        self.assertEqual(calls[1]['feedback'][0]['workflow']['next_action']['execution']['status'],
                         'launch_failed')
        self.assertEqual(result['rounds'],2)

    def test_interrupted_budget_is_not_granted_again(self):
        with patch('precision_rsi.evolve.improve',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run(self.config,self.root/'session')
        state=read(self.root/'session/evolution.json')
        self.assertGreaterEqual(state['seconds'],5)
        self.cfg['budget_seconds']=5;self.save()
        # Binding changes are rejected even when no round was finished.
        with self.assertRaisesRegex(ValueError,'changed'):
            run(self.config,self.root/'session')

    def test_acceptance_feeds_the_following_round(self):
        calls=[]
        def improve(config,workspace,**kwargs):
            calls.append(read(config))
            if len(calls)==1:
                return dict(status='completed',candidates=[],evaluation=dict(evaluation=dict(accepted=True)))
            return dict(status='completed',candidates=[],no_update=True)
        with patch('precision_rsi.evolve.improve',side_effect=improve):
            result=run(self.config,self.root/'session')
        self.assertEqual(result['accepted'],1)
        self.assertTrue(calls[1]['feedback'][0]['accepted'])

    def test_total_budget_includes_evaluation_time(self):
        clock=[100.0]
        def improve(config,workspace,**kwargs):
            self.assertLessEqual(kwargs['time_budget_seconds'],5)
            clock[0]+=31
            return dict(status='completed',candidates=[])
        with patch('precision_rsi.evolve.time.monotonic',side_effect=lambda:clock[0]), \
                patch('precision_rsi.evolve.improve',side_effect=improve) as agent:
            result=run(self.config,self.root/'session')
        self.assertEqual(agent.call_count,1)
        self.assertEqual(result['reason'],'budget exhausted')
        self.assertEqual(result['spent_seconds'],31)

    def test_feedback_separates_build_cost_from_task_measurements(self):
        measurement=dict(baseline=dict(seconds=100,total_tokens=1000),
            candidate=dict(seconds=110,total_tokens=1100,usage=dict(total_tokens=500)),
            construction=dict(seconds=60,usage=dict(total_tokens=600)),amortize_over_tasks=1)
        result=dict(status='completed',candidates=[],evaluation=dict(evaluation=dict(
            accepted=False,measurement=measurement)))
        feedback,_=_feedback(result,self.root/'memory')
        received=feedback['measurements']
        self.assertEqual(received['candidate']['seconds']-received['construction']['seconds'],50)
        self.assertEqual(received['candidate']['usage']['total_tokens'],500)
        self.assertEqual(received['amortize_over_tasks'],1)
        self.assertFalse(feedback['accepted'])

    def test_recovered_tool_feedback_keeps_interfaces_costs_and_native_diagnostics(self):
        tool=self.root/'tool.py';tool.write_text('def compare(a,b): return a==b\n')
        store=self.root/'memory'
        identity=propose(store,dict(kind='tool',content='compare(a,b) -> bool',tags=['compare'],
            artifacts=[str(tool)]),{})
        native=dict(status='needs_action',reason='selection mismatch',
            selection=dict(consumer=dict(tags=['compare'],limit_chars=20)))
        logs=dict(stdout=dict(tail='{"type":"thinking","text":"FULL_TRACE_MARKER"}\n'+json.dumps(native)),
                  stderr=dict(tail='verbose traceback'))
        workflow=dict(status='needs_action',next_action=dict(stage='evaluate',reason='native command failed',
            execution=dict(status='failed',logs=logs)))
        result=dict(status='completed',candidates=[identity],spent_seconds=12,usage=dict(total_tokens=34),
            proposal_recovery=dict(status='timed_out',path='/retained/proposals.json'),workflow=workflow)
        feedback,_=_feedback(result,store)
        self.assertEqual(feedback['updates'][0]['artifacts'].keys(),{str(tool)})
        self.assertEqual(feedback['construction'],dict(seconds=12,usage=dict(total_tokens=34)))
        self.assertEqual(feedback['proposal_recovery']['status'],'timed_out')
        self.assertEqual(feedback['workflow']['next_action']['diagnostic'],native)
        self.assertNotIn('FULL_TRACE_MARKER',json.dumps(feedback))
        self.assertNotIn('verbose traceback',json.dumps(feedback))


if __name__=='__main__':
    unittest.main()
