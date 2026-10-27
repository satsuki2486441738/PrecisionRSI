"""External controller feedback, stopping and replay regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from precision_rsi.evolve import run
from precision_rsi.memory import read


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

    def test_repeated_candidate_stops(self):
        with patch('precision_rsi.evolve.improve',return_value=dict(status='completed',candidates=[])) as agent:
            result=run(self.config,self.root/'session')
        self.assertEqual(agent.call_count,2)
        self.assertEqual(result['reason'],'repeated candidate')

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


if __name__=='__main__':
    unittest.main()
