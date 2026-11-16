"""External controller feedback, stopping and replay regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from precision_rsi.evolve import _feedback, run
from precision_rsi.memory import propose, read, write
from precision_rsi.snapshots import Journal


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
        journal = Journal(self.root/'session/.snapshots')
        initial = journal.event('checkpoint-000')
        first = journal.event('round-001-committed')
        second = journal.event('round-002-committed')
        self.assertTrue(initial['checkpoint'])
        self.assertTrue(first['checkpoint'])  # Rejected/failed work still changes research state.
        self.assertNotEqual(first['id'], second['id'])
        self.assertEqual(journal.event('round-002-start')['parent'], first['id'])
        final = journal.event('run-completed')
        self.assertEqual(final['parent'], second['id'])
        self.assertEqual(read(self.root/'session/evolution.json')['checkpoint'], final['id'])
        terminal_state = read(journal.root/'objects'/final['files']['audit/controller.json']['object'])
        self.assertEqual(terminal_state['result'], result)
        count = len(list((journal.root/'events').glob('*.json')))
        with patch('precision_rsi.evolve.improve',side_effect=AssertionError('replayed model')):
            self.assertEqual(run(self.config,self.root/'session'),result)
        self.assertEqual(len(list((journal.root/'events').glob('*.json'))), count)

    def test_terminal_decision_is_preserved_at_the_round_limit(self):
        self.cfg['max_rounds'] = 1
        self.save()
        with patch('precision_rsi.evolve.improve', return_value=dict(
                status='completed', candidates=[], no_update=True)):
            result = run(self.config, self.root/'last-round')
        self.assertEqual(result['reason'], 'no useful update')

    def test_repeated_failure_stops(self):
        with patch('precision_rsi.evolve.improve',return_value=dict(status='completed',candidates=[])) as agent:
            result=run(self.config,self.root/'session')
        self.assertEqual(agent.call_count,2)
        self.assertEqual(result['reason'],'repeated failure')

    def test_sealed_terminal_recovers_interrupted_state_write_without_agent(self):
        with patch('precision_rsi.evolve.improve',return_value=dict(status='completed',candidates=[],no_update=True)):
            result = run(self.config, self.root/'session')
        path = self.root/'session/evolution.json'
        state = read(path)
        state.pop('result')
        state['rounds'] = []
        state['reservation'] = 5
        write(path, state)
        with patch('precision_rsi.evolve.improve',side_effect=AssertionError('repeated agent work')):
            self.assertEqual(run(self.config,self.root/'session'), result)
        self.assertEqual(read(path)['result'], result)

    def test_committed_terminal_round_recovers_without_extra_agent_call(self):
        real_write = write
        workspace = self.root/'commit-crash'

        def crash(path, value):
            event = workspace/'.snapshots/events/round-001-committed.json'
            if event.exists() and value.get('checkpoint') == read(event)['id'] and not value.get('result'):
                raise KeyboardInterrupt('crash before controller state write')
            real_write(path, value)

        with patch('precision_rsi.evolve.improve', return_value=dict(
                status='completed', candidates=[], no_update=True)):
            with patch('precision_rsi.evolve.write', side_effect=crash):
                with self.assertRaises(KeyboardInterrupt):
                    run(self.config, workspace)
        with patch('precision_rsi.evolve.improve', side_effect=AssertionError('extra model call')):
            result = run(self.config, workspace)
        self.assertEqual(result['rounds'], 1)
        self.assertEqual(result['reason'], 'no useful update')

    def test_unfinished_researcher_does_not_consume_another_model_round(self):
        def interrupted(config, workspace, **kwargs):
            from precision_rsi.research import Research
            info = read(config)['research']
            research = Research(info['root'], info['binding'])
            research.begin(info['turn'], {'input':'started'})
            raise KeyboardInterrupt()
        workspace = self.root/'unfinished'
        with patch('precision_rsi.evolve.improve', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                run(self.config, workspace)
        with patch('precision_rsi.evolve.improve', side_effect=AssertionError('blind retry')):
            result = run(self.config, workspace)
        self.assertEqual(result['status'], 'needs_action')
        self.assertIn('reconcile', result['reason'])

    def test_runtime_binds_compaction_policy_and_skill_references(self):
        from precision_rsi import versions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'precision_rsi').mkdir()
            policy = root/'precision_rsi/compact.json'
            policy.write_text('first')
            reference = root/'skills/compare/references/guide.md'
            reference.parent.mkdir(parents=True)
            reference.write_text('guide')
            with patch.object(versions, '__file__', str(root/'precision_rsi/versions.py')):
                before = versions._runtime()
                policy.write_text('second')
                after = versions._runtime()
        self.assertIn('skills/compare/references/guide.md', before)
        self.assertNotEqual(before, after)

    def test_improver_launch_failure_stops_and_replay_does_not_retry(self):
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
        self.assertEqual(len(calls),1)
        self.assertEqual(result['rounds'],1)
        self.assertEqual(result['status'],'needs_action')
        self.assertEqual(result['reason'],'infrastructure failure')
        with patch('precision_rsi.evolve.improve',side_effect=AssertionError('infrastructure retry')):
            self.assertEqual(run(self.config,self.root/'launch-failure'),result)

    def test_native_harness_error_stops_without_counting_unknown_usage_as_zero(self):
        native=dict(type='status',phase='turn_end',reason=dict(kind='error',error=dict(
            code='MISSING_CREDENTIAL',message='private credential detail')))
        folder=self.root/'native';folder.mkdir()
        (folder/'stdout.bin').write_text(json.dumps(native)+'\n'+'x'*3000)
        workflow=dict(status='needs_action',next_action=dict(stage='improve',run=str(folder),
            execution=dict(status='failed',logs=dict(stdout=dict(tail='x'*1000)))))
        value=dict(status='needs_action',workflow=workflow,candidates=[],usage=None)
        with patch('precision_rsi.evolve.improve',return_value=value) as agent:
            result=run(self.config,self.root/'native-failure')
        self.assertEqual(agent.call_count,1)
        self.assertEqual(result['reason'],'infrastructure failure')
        self.assertIsNone(result['usage']['total_tokens'])
        feedback=read(self.root/'native-failure/evolution.json')['rounds'][0]['feedback']
        self.assertEqual(feedback['infrastructure'],dict(kind='harness_error',code='MISSING_CREDENTIAL'))

    def test_task_failure_is_not_an_improver_infrastructure_failure(self):
        failed=dict(status='needs_action',next_action=dict(execution=dict(status='launch_failed')))
        value=dict(candidates=[],evaluation=dict(evaluation=dict(workflow=failed)))
        feedback,_=_feedback(value,self.root/'memory')
        self.assertNotIn('infrastructure',feedback)

    def test_harness_terminal_status_is_checked_even_with_zero_exit(self):
        native=dict(type='status',phase='turn_end',reason=dict(kind='error',error=dict(code='MISSING_CREDENTIAL')))
        execution=dict(status='completed',returncode=0,logs=dict(stdout=dict(tail=json.dumps(native))))
        value=dict(no_update=True,workflow=dict(status='completed',stages=[dict(execution=execution)]))
        with patch('precision_rsi.evolve.improve',return_value=value) as agent:
            result=run(self.config,self.root/'zero-exit-error')
        self.assertEqual(agent.call_count,1)
        self.assertEqual(result['reason'],'infrastructure failure')
        self.assertEqual(result['status'],'needs_action')
        # A later successful native terminal event supersedes a recovered error.
        execution['logs']['stdout']['tail']+='\n'+json.dumps(dict(type='status',phase='turn_end',reason=dict(kind='completed')))
        feedback,_=_feedback(value,self.root)
        self.assertNotIn('infrastructure',feedback)

    def test_interrupted_budget_is_not_granted_again(self):
        with patch('precision_rsi.evolve.improve',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run(self.config,self.root/'session')
        state=read(self.root/'session/evolution.json')
        journal = Journal(self.root/'session/.snapshots')
        self.assertFalse(journal.event('round-001-start')['checkpoint'])
        self.assertFalse((journal.root/'events/round-001-committed.json').exists())
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

    def test_raw_usage_includes_evaluation_without_recounting_construction(self):
        construction=dict(input_tokens=10,output_tokens=2,total_tokens=12)
        measured=dict(input_tokens=20,output_tokens=3,total_tokens=23)
        value=dict(status='completed',no_update=True,candidates=[],usage=construction,
                   evaluation=dict(evaluation=dict(accepted=False,measurement=dict(usage=measured))))
        with patch('precision_rsi.evolve.improve',return_value=value):
            result=run(self.config,self.root/'usage')
        self.assertEqual(result['usage']['total_tokens'],35)
        self.assertEqual(result['construction_usage']['total_tokens'],12)
        self.assertEqual(result['evaluation_usage']['total_tokens'],23)
        with patch('precision_rsi.evolve.improve',side_effect=AssertionError('replayed call')):
            self.assertEqual(run(self.config,self.root/'usage'),result)

    def test_unmeasured_evaluation_usage_is_unknown(self):
        value=dict(status='completed',no_update=True,candidates=[],usage=dict(total_tokens=12),
                   evaluation=dict(evaluation=dict(accepted=False,measurement=None)))
        with patch('precision_rsi.evolve.improve',return_value=value):
            result=run(self.config,self.root/'unknown-usage')
        self.assertIsNone(result['evaluation_usage']['total_tokens'])
        self.assertIsNone(result['usage']['total_tokens'])
        self.assertEqual(result['construction_usage']['total_tokens'],12)

    def test_feedback_separates_build_cost_from_task_measurements(self):
        measurement=dict(baseline=dict(seconds=100,total_tokens=1000),
            candidate=dict(seconds=80,total_tokens=800,usage=dict(total_tokens=500),
                execution=dict(seconds=50,total_tokens=500,usage=dict(total_tokens=500)),
                including_construction=dict(seconds=80,total_tokens=800,cost=None)),
            construction=dict(seconds=60,usage=dict(total_tokens=600)),amortize_over_tasks=2)
        result=dict(status='completed',candidates=[],evaluation=dict(evaluation=dict(
            accepted=False,measurement=measurement)))
        feedback,_=_feedback(result,self.root/'memory')
        received=feedback['measurements']
        self.assertEqual(received['candidate']['execution']['seconds'],50)
        self.assertEqual(received['candidate']['execution']['usage']['total_tokens'],500)
        self.assertEqual(received['candidate']['including_construction']['seconds'],80)
        self.assertNotIn('seconds',received['candidate'])
        self.assertEqual(received['amortize_over_tasks'],2)
        self.assertFalse(feedback['accepted'])

    def test_legacy_feedback_does_not_guess_execution_time(self):
        measurement=dict(candidate=dict(seconds=110,usage=dict(total_tokens=500)),
                         construction=dict(seconds=60),amortize_over_tasks=1)
        received,_=_feedback(dict(evaluation=dict(evaluation=dict(measurement=measurement))),self.root)
        candidate=received['measurements']['candidate']
        self.assertIsNone(candidate['execution']['seconds'])
        self.assertEqual(candidate['execution']['total_tokens'],500)
        self.assertEqual(candidate['including_construction']['seconds'],110)

    def test_selection_preflight_feedback_retains_exact_rejections(self):
        selection=dict(missing_updates=['tool'],cases=[dict(case='public',selected=[],
            rejected=[dict(id='tool',reason='interface_budget',interface_chars=500,remaining_chars=10)])])
        feedback,_=_feedback(dict(candidates=[],selection=selection),self.root)
        self.assertEqual(feedback['selection'],selection)

    def test_feedback_does_not_label_evaluation_time_as_construction(self):
        construction=dict(seconds=480,usage=dict(total_tokens=612418),cost=None)
        result=dict(candidates=[],spent_seconds=900,usage=construction['usage'],
                    construction=construction)
        feedback,_=_feedback(result,self.root/'memory')
        self.assertEqual(feedback['construction'],construction)
        self.assertNotEqual(feedback['construction']['seconds'],result['spent_seconds'])

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
        self.assertEqual(feedback['updates'][0]['artifacts'],[str(tool)])
        # A retained interface can be submitted unchanged instead of copying the
        # host's fingerprint mapping into the path-array proposal field.
        self.assertEqual(propose(store,feedback['updates'][0],{}),identity)
        self.assertEqual(feedback['construction'],dict(seconds=12,usage=dict(total_tokens=34)))
        self.assertEqual(feedback['proposal_recovery']['status'],'timed_out')
        self.assertEqual(feedback['workflow']['next_action']['diagnostic'],native)
        self.assertNotIn('FULL_TRACE_MARKER',json.dumps(feedback))
        self.assertNotIn('verbose traceback',json.dumps(feedback))


if __name__=='__main__':
    unittest.main()
