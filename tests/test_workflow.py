"""Native-command regressions for cross-skill reuse, invalidation and continuation."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'skills/precision-compare/scripts'
sys.path.insert(0, str(TOOLS))
from workflow import inspect_workflow, run_workflow


@unittest.skipUnless(os.name == 'posix', 'POSIX workflow lock')
class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / 'session'
        self.plan_path = self.root / 'task.json'
        (self.root / 'environment.json').write_text('{"runtime":"fixture-v1"}')
        (self.root / 'factor.json').write_text('1')
        (self.root / 'data.json').write_text('[1, 3]')
        (self.root / 'helper.py').write_text('''import json,sys
from pathlib import Path
kind,out=sys.argv[1],Path(sys.argv[2])
if kind=='adapt':
 result={'status':'ready','environment':json.loads(Path('environment.json').read_text())}
elif kind=='collect':
 factor=json.loads(Path('factor.json').read_text())
 result={'values':[factor*x for x in json.loads(Path('data.json').read_text())]}
else:
 values=json.loads(Path(sys.argv[3]).read_text())['values']
 expected=json.loads(Path('data.json').read_text())
 result={'correct':values==expected,'max_abs':max(abs(a-b) for a,b in zip(values,expected))}
out.write_text(json.dumps(result))
print(kind)
''')
        self.spec = dict(schema_version=1, id='fixture', budget_seconds=30, stages=[
            dict(id='adapt', skill='framework-adapt', inputs=['helper.py', 'environment.json'],
                 argv=[sys.executable, 'helper.py', 'adapt', '{run}/environment.json'],
                 outputs={'environment': 'environment.json'}, timeout=5,
                 checks=[dict(output='environment', pointer='/status', equals='ready')]),
            dict(id='collect', skill='precision-compare', depends_on=['adapt'],
                 inputs=['helper.py', 'factor.json', 'data.json'],
                 argv=[sys.executable, 'helper.py', 'collect', '{run}/capture.json'],
                 outputs={'capture': 'capture.json'}, timeout=5),
            dict(id='compare', skill='precision-compare', depends_on=['collect'],
                 inputs=['helper.py', 'data.json'],
                 argv=[sys.executable, 'helper.py', 'compare', '{run}/metrics.json', '{artifact:collect:capture}'],
                 outputs={'metrics': 'metrics.json'}, timeout=5,
                 checks=[dict(output='metrics', pointer='/correct', equals=True)],
                 on_check_failure='precision-repair')])
        self.save()

    def save(self):
        self.plan_path.write_text(json.dumps(self.spec))

    def run_task(self, refresh=()):
        return run_workflow(self.plan_path, self.workspace, refresh)

    def execution_plan(self, seed):
        (self.root/'execution-seed.json').write_text(json.dumps(seed))
        (self.root/'native.py').write_text(
            'import json,os,sys\nfrom pathlib import Path\n'
            'assert os.environ["NATIVE_READY"]=="yes", "native environment not ready"\n'
            'Path(sys.argv[1]).write_text(json.dumps({"ready":True,"path":os.environ["PATH"]}))\n')
        self.spec['stages'] = [
            dict(id='adapt', skill='framework-adapt', timeout=5,
                 inputs=['execution-seed.json'], outputs={'execution':'execution.json'},
                 argv=[sys.executable,'-c',
                       'import shutil,sys;shutil.copyfile("execution-seed.json",sys.argv[1])',
                       '{run}/execution.json']),
            dict(id='preflight', skill='framework-adapt', timeout=5,
                 depends_on=['adapt'], inputs=['native.py'],
                 execution='{artifact:adapt:execution}',
                 argv=['native.py','{run}/ready.json'], outputs={'ready':'ready.json'},
                 checks=[dict(output='ready', pointer='/ready', equals=True)]),
            dict(id='capture', skill='precision-compare', timeout=5,
                 depends_on=['adapt','preflight'], inputs=['native.py'],
                 execution='{artifact:adapt:execution}',
                 argv=['native.py','{run}/capture.json'], outputs={'capture':'capture.json'})]
        self.save()

    def test_adaptation_execution_artifact_is_shared_by_preflight_and_capture(self):
        self.execution_plan(dict(argv_prefix=[sys.executable],env={'NATIVE_READY':'yes'},
                                 path_prepend=[str(self.root)]))
        result=self.run_task()
        self.assertEqual(result['status'],'completed')
        for stage in result['stages'][1:]:
            command=json.loads((Path(stage['run'])/'command.json').read_text())
            self.assertEqual(command['argv'][0],sys.executable)
            self.assertEqual(command['execution']['path'],
                result['stages'][0]['outputs']['execution']['path'])
        capture=json.loads(Path(result['stages'][2]['outputs']['capture']['path']).read_text())
        self.assertTrue(capture['path'].startswith(str(self.root)+os.pathsep))
        with patch('workflow.record_run',side_effect=AssertionError('unexpected launch')):
            self.assertEqual(inspect_workflow(self.plan_path,self.workspace)['status'],'completed')
            replay=self.run_task()
        self.assertTrue(all(stage['reused'] for stage in replay['stages']))

    def test_wrapped_executable_change_invalidates_cached_capture(self):
        native=self.root/'native-executable'
        def version(text):
            native.write_text('#!'+sys.executable+'\nimport sys\nfrom pathlib import Path\n'
                'Path(sys.argv[1]).write_text('+repr(text)+')\n')
            native.chmod(0o755)
        version('old')
        (self.root/'execution.json').write_text(json.dumps(dict(argv_prefix=['/usr/bin/env'])))
        self.spec['stages']=[dict(id='capture',skill='precision-compare',timeout=5,
            execution='execution.json',argv=[str(native),'{run}/capture.txt'],
            outputs={'capture':'capture.txt'})]
        self.save();first=self.run_task();version('new');changed=self.run_task()
        self.assertFalse(changed['stages'][0]['reused'])
        self.assertEqual(Path(changed['stages'][0]['outputs']['capture']['path']).read_text(),'new')
        self.assertEqual(Path(first['stages'][0]['outputs']['capture']['path']).read_text(),'old')

    def test_failed_output_validation_includes_tails_with_native_zero_status(self):
        (self.root/'factor.json').write_text('2')
        result=self.run_task()
        command=result['next_action']['execution']
        self.assertEqual(command['status'],'completed')
        self.assertEqual(command['returncode'],0)
        self.assertIn('compare',command['logs']['stdout']['tail'])

    def test_scheduler_allocation_change_invalidates_native_receipts(self):
        with patch.dict(os.environ,{'NVIDIA_VISIBLE_DEVICES':'GPU-first'}):
            first=self.run_task()
            replay=self.run_task()
        self.assertTrue(all(s['reused'] for s in replay['stages']))
        with patch.dict(os.environ,{'NVIDIA_VISIBLE_DEVICES':'GPU-second'}):
            changed=self.run_task()
        self.assertEqual(changed['status'],'completed')
        self.assertFalse(any(s['reused'] for s in changed['stages']))
        self.assertEqual(changed['attempts'],first['attempts']+3)

    def test_native_preflight_failure_blocks_capture_and_retains_failure(self):
        self.execution_plan(dict(argv_prefix=[sys.executable],env={'NATIVE_READY':'no'}))
        failed=self.run_task()
        self.assertEqual(failed['status'],'needs_action')
        self.assertEqual(failed['attempts'],2)
        self.assertEqual(failed['next_action']['stage'],'preflight')
        stderr=Path(failed['next_action']['stderr']).read_text()
        self.assertIn('native environment not ready',stderr)
        self.assertIn('native environment not ready',failed['next_action']['execution']['logs']['stderr']['tail'])
        with patch('workflow.record_run',side_effect=AssertionError('retrying unchanged failure')):
            self.assertEqual(self.run_task()['attempts'],2)
        (self.root/'execution-seed.json').write_text(json.dumps(
            dict(argv_prefix=[sys.executable],env={'NATIVE_READY':'yes'})))
        fixed=self.run_task()
        self.assertEqual(fixed['status'],'completed')
        self.assertEqual(fixed['attempts'],5)
        self.assertEqual(Path(failed['next_action']['stderr']).read_text(),stderr)

    def test_execution_file_change_invalidates_only_consumers_and_descendants(self):
        profile=self.root/'execution.json'
        profile.write_text(json.dumps(dict(env={'PRECISION_SETTING':'one'})))
        self.spec['stages'][1]['execution']='execution.json'
        self.save();first=self.run_task()
        profile.write_text(json.dumps(dict(env={'PRECISION_SETTING':'two'})))
        changed=self.run_task()
        self.assertEqual(changed['status'],'completed')
        self.assertEqual([s['reused'] for s in changed['stages']],[True,False,False])
        self.assertEqual(first['stages'][0]['run'],changed['stages'][0]['run'])

    def test_adapt_compare_repair_revalidation_and_zero_command_replay(self):
        (self.root / 'factor.json').write_text('2')
        first = self.run_task()
        self.assertEqual(first['status'], 'needs_action')
        self.assertEqual(first['next_action']['skill'], 'precision-repair')
        original_metrics = Path(first['next_action']['outputs']['metrics']['path'])
        original_bytes = original_metrics.read_bytes()
        self.assertEqual(json.loads(original_bytes)['max_abs'], 3)
        repeated_failure = self.run_task()
        self.assertEqual(repeated_failure['attempts'], first['attempts'])
        (self.root / 'factor.json').write_text('1')
        repaired = self.run_task()
        self.assertEqual(repaired['status'], 'completed')
        self.assertEqual([s['reused'] for s in repaired['stages']], [True, False, False])
        self.assertEqual(original_metrics.read_bytes(), original_bytes)
        replay = self.run_task()
        self.assertEqual(replay['attempts'], repaired['attempts'])
        self.assertTrue(all(s['reused'] for s in replay['stages']))
        self.assertEqual(replay['spent_seconds'], repaired['spent_seconds'])
        self.assertEqual(inspect_workflow(self.plan_path, self.workspace)['status'], 'completed')

    def test_input_and_output_changes_invalidate_affected_dependency_chain(self):
        original = self.run_task()
        runs = [s['run'] for s in original['stages']]
        # Corrupted capture bytes must be recollected, even with unchanged inputs.
        Path(original['stages'][1]['outputs']['capture']['path']).write_text('{"values":[99]}')
        current = self.run_task()
        self.assertEqual(current['status'], 'completed')
        self.assertEqual(current['stages'][0]['run'], runs[0])
        self.assertNotEqual(current['stages'][1]['run'], runs[1])
        # The regenerated bytes are identical to the original, so the valid metric
        # artifact can be reused instead of repeating arithmetic unnecessarily.
        self.assertTrue(current['stages'][2]['reused'])
        (self.root / 'environment.json').write_text('{"runtime":"fixture-v2"}')
        changed = self.run_task()
        self.assertFalse(any(s['reused'] for s in changed['stages']))

    def test_explicit_refresh_reruns_descendants_and_preserves_other_runs(self):
        first = self.run_task()
        result = self.run_task(refresh=['collect'])
        self.assertEqual([s['reused'] for s in result['stages']], [True, False, False])
        self.assertEqual(result['attempts'], first['attempts'] + 2)
        self.assertTrue(all(Path(s['run']).exists() for s in first['stages']))

    def test_followup_review_change_and_fresh_measurement_preserve_valid_work(self):
        # A sibling observation is independent of the requested measurement.
        (self.root / 'sibling.json').write_text('{"observation":7}')
        self.spec['stages'].append(dict(id='sibling', skill='precision-compare',
            inputs=['sibling.json'], depends_on=['adapt'], timeout=5,
            argv=[sys.executable, '-c',
                  'from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(Path("sibling.json").read_bytes())',
                  '{run}/observation.json'], outputs={'observation': 'observation.json'}))
        self.save()
        first = self.run_task()
        self.assertEqual(first['attempts'], 4)
        originals = {v['path']: Path(v['path']).read_bytes()
                     for stage in first['stages'] for v in stage['outputs'].values()}
        state_bytes = (self.workspace / 'state.json').read_bytes()
        with patch('workflow.record_run', side_effect=AssertionError('unexpected native execution')):
            review = inspect_workflow(self.plan_path, self.workspace)
            self.assertEqual((self.workspace / 'state.json').read_bytes(), state_bytes)
            repeated = self.run_task()
        for result in (review, repeated):
            self.assertEqual(result['attempts'], 4)
            self.assertEqual(result['spent_seconds'], first['spent_seconds'])
            self.assertTrue(all(stage['reused'] for stage in result['stages']))
        (self.root / 'data.json').write_text('[2, 4]')
        changed = self.run_task()
        self.assertEqual(changed['attempts'], 6)
        self.assertEqual([stage['reused'] for stage in changed['stages']], [True, False, False, True])
        fresh = self.run_task(refresh=['collect'])
        self.assertEqual(fresh['attempts'], 8)
        self.assertEqual([stage['reused'] for stage in fresh['stages']], [True, False, False, True])
        self.assertNotEqual(fresh['stages'][1]['run'], changed['stages'][1]['run'])
        self.assertNotEqual(fresh['stages'][2]['run'], changed['stages'][2]['run'])
        self.assertEqual(fresh['stages'][0]['run'], first['stages'][0]['run'])
        self.assertEqual(fresh['stages'][3]['run'], first['stages'][3]['run'])
        for path, content in originals.items():
            self.assertEqual(Path(path).read_bytes(), content)

    def test_comparison_implementation_change_reuses_environment_and_captures(self):
        comparator = self.root / 'comparator.py'
        comparator.write_bytes((self.root / 'helper.py').read_bytes())
        self.spec['stages'][2]['argv'][1] = 'comparator.py'
        self.spec['stages'][2]['inputs'] = ['comparator.py', 'data.json']
        self.save()
        first = self.run_task()
        captures = first['stages'][1]['outputs']['capture']
        original_bytes = Path(captures['path']).read_bytes()
        comparator.write_text(comparator.read_text().replace(
            'out.write_text(json.dumps(result))',
            'result["comparison_version"]=2;out.write_text(json.dumps(result))'))
        changed = self.run_task()
        self.assertEqual(changed['attempts'], first['attempts'] + 1)
        self.assertEqual([stage['reused'] for stage in changed['stages']], [True, True, False])
        self.assertEqual(changed['stages'][1]['outputs']['capture'], captures)
        self.assertEqual(Path(captures['path']).read_bytes(), original_bytes)
        metrics = json.loads(Path(changed['stages'][2]['outputs']['metrics']['path']).read_text())
        self.assertTrue(metrics['correct'])
        self.assertEqual(metrics['comparison_version'], 2)

    def test_live_probe_can_repeat_without_repeating_unchanged_downstream_work(self):
        self.spec['stages'][0]['cache'] = False
        self.save()
        first = self.run_task()
        result = self.run_task()
        self.assertEqual([s['reused'] for s in result['stages']], [False, True, True])
        self.assertEqual(result['attempts'], first['attempts'] + 1)

    def test_selected_environment_change_invalidates_only_its_stage_and_descendants(self):
        self.spec['stages'][1]['env_keys'] = ['PRECISION_TEST_SETTING']
        self.save()
        with patch.dict(os.environ, {'PRECISION_TEST_SETTING': 'first'}):
            self.run_task()
        with patch.dict(os.environ, {'PRECISION_TEST_SETTING': 'second'}):
            current = self.run_task()
        self.assertEqual([s['reused'] for s in current['stages']], [True, False, False])

    def test_workload_identity_is_shared_and_changes_invalidate_prior_stages(self):
        self.spec['workload'] = dict(model='tiny', task='sft', sides=['amber','blue'])
        self.save()
        first = self.run_task()
        self.assertEqual(first['workload'], self.spec['workload'])
        self.spec['workload']['model'] = 'different'
        self.save()
        current = self.run_task()
        self.assertFalse(any(s['reused'] for s in current['stages']))
        state = json.loads((self.workspace / 'state.json').read_text())
        self.assertEqual(state['history'][0]['identity']['workload']['model'], 'tiny')
        self.assertEqual(state['history'][-1]['identity']['workload']['model'], 'different')

    def test_execution_tool_update_invalidates_previous_validation(self):
        implementation = self.root / 'runner-version.py'
        implementation.write_text('version-one')
        with patch('workflow.ENGINE_FILES', (implementation,)):
            self.run_task()
            self.assertTrue(all(s['reused'] for s in self.run_task()['stages']))
            implementation.write_text('version-two-with-new-validation')
            current = self.run_task()
        self.assertFalse(any(s['reused'] for s in current['stages']))

    def test_status_is_read_only_and_missing_sources_route_to_owner(self):
        status = inspect_workflow(self.plan_path, self.workspace)
        self.assertEqual(status['next_action']['skill'], 'framework-adapt')
        self.assertFalse(self.workspace.exists())
        (self.root / 'helper.py').unlink()
        status = self.run_task()
        self.assertIn('missing input', status['next_action']['reason'])
        self.assertEqual(status['attempts'], 0)

    def test_invalid_plans_fail_before_execution(self):
        for change in ('cycle', 'unknown-artifact', 'escape', 'zero-budget', 'wrong-skill'):
            with self.subTest(change=change):
                spec = json.loads(json.dumps(self.spec))
                if change == 'cycle':
                    spec['stages'][0]['depends_on'] = ['compare']
                elif change == 'unknown-artifact':
                    spec['stages'][2]['argv'][-1] = '{artifact:adapt:missing}'
                elif change == 'escape':
                    spec['stages'][0]['outputs']['environment'] = '../external.json'
                elif change == 'zero-budget':
                    spec['budget_seconds'] = 0
                else:
                    spec['stages'][0]['skill'] = 'unknown'
                self.plan_path.write_text(json.dumps(spec))
                with self.assertRaises(ValueError):
                    self.run_task()
                self.assertFalse(self.workspace.exists())

    def test_exit_zero_without_outputs_never_becomes_a_cache_hit(self):
        self.spec['stages'] = [dict(id='empty', skill='framework-adapt', timeout=5,
                                   inputs=[], argv=[sys.executable, '-c', 'print("no output")'],
                                   outputs={'missing': 'missing.json'})]
        self.save()
        first = self.run_task()
        self.assertEqual(first['next_action']['skill'], 'framework-adapt')
        self.assertIn('missing input/output', first['next_action']['reason'])
        self.assertEqual(self.run_task()['attempts'], 1)

    def test_cumulative_budget_includes_prior_commands_and_retries(self):
        first = self.run_task()
        self.spec['budget_seconds'] = first['spent_seconds']
        self.save()
        (self.root / 'data.json').write_text('[2, 4]')
        current = self.run_task()
        self.assertEqual(current['attempts'], first['attempts'])
        self.assertIn('budget exhausted', current['next_action']['reason'])
        self.spec['budget_seconds'] += 10
        self.save()
        finished = self.run_task()
        self.assertEqual(finished['status'], 'completed')
        self.assertGreater(finished['spent_seconds'], first['spent_seconds'])

    def test_inputs_changing_during_execution_are_not_verified(self):
        self.spec['stages'] = [dict(id='change', skill='framework-adapt', timeout=5,
            inputs=['factor.json'], outputs={'result': 'result.json'}, argv=[sys.executable, '-c',
            'from pathlib import Path; import sys; Path("factor.json").write_text("3"); Path(sys.argv[1]).write_text("{}")', '{run}/result.json'])]
        self.save()
        report = self.run_task()
        self.assertEqual(report['status'], 'needs_action')
        self.assertIn('inputs changed', report['next_action']['reason'])
        self.assertEqual(report['next_action']['skill'], 'framework-adapt')

    def test_zero_exit_with_wrong_json_semantics_routes_to_repair(self):
        self.spec['stages'] = [dict(id='check', skill='precision-compare', timeout=5,
            inputs=[], outputs={'result': 'result.json'}, on_check_failure='precision-repair',
            checks=[dict(output='result', pointer='/ok', equals=True)],
            argv=[sys.executable, '-c', 'import sys;open(sys.argv[1],"w").write(\'{"ok":1}\')', '{run}/result.json'])]
        self.save()
        result = self.run_task()
        self.assertEqual(result['next_action']['skill'], 'precision-repair')
        self.assertIn('output check failed', result['next_action']['reason'])

    def test_native_failure_routes_to_owner_instead_of_assuming_a_precision_defect(self):
        self.spec['stages'] = [dict(id='native', skill='precision-compare', timeout=5,
            inputs=[], outputs={'result': 'result.json'}, on_check_failure='precision-repair',
            argv=[sys.executable, '-c', 'raise RuntimeError("collector failed")'])]
        self.save()
        report = self.run_task()
        self.assertEqual(report['next_action']['skill'], 'precision-compare')
        self.assertIn('failed', report['next_action']['reason'])
        self.assertIn('collector failed', Path(report['next_action']['stderr']).read_text())

    def test_downstream_mutation_of_dependency_artifact_is_rejected(self):
        self.spec['stages'][1]['argv'] = [sys.executable, '-c',
            'import sys;open(sys.argv[1],"w").write("changed");open(sys.argv[2],"w").write("{}")',
            '{artifact:adapt:environment}', '{run}/capture.json']
        self.save()
        report = self.run_task()
        self.assertEqual(report['status'], 'needs_action')
        self.assertIn('dependency output changed', report['next_action']['reason'])

    def test_malformed_or_missing_capture_fields_route_to_collector_owner(self):
        for index, data in enumerate(('{}', 'not JSON')):
            with self.subTest(data=data):
                self.spec['stages'] = [dict(id='format'+str(index), skill='precision-compare', timeout=5,
                    inputs=[], outputs={'result': 'result.json'}, on_check_failure='precision-repair',
                    checks=[dict(output='result', pointer='/correct', equals=True)],
                    argv=[sys.executable, '-c', 'import sys;open(sys.argv[1],"w").write(sys.argv[2])',
                          '{run}/result.json', data])]
                self.save()
                report = self.run_task()
                self.assertEqual(report['status'], 'needs_action')
                self.assertEqual(report['next_action']['skill'], 'precision-compare')

    def test_lock_prevents_duplicate_execution(self):
        import fcntl
        self.workspace.mkdir()
        with (self.workspace / 'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, 'active runner'):
                self.run_task()
        self.assertFalse((self.workspace / 'state.json').exists())

    def test_orphaned_native_child_prevents_duplicate_execution(self):
        self.workspace.mkdir()
        folder = self.workspace / 'runs/compare-fixture-adapt-attempt1-20261022T000000Z'
        folder.mkdir(parents=True)
        (folder / 'command.json').write_text(json.dumps(dict(status='running', child_pid=os.getpid())))
        state = dict(schema_version=1, id='fixture', root=str(self.root.resolve()), history=[dict(
            stage='adapt', status='running', key='unknown', run_prefix='compare-fixture-adapt-attempt1-',
            timeout=5, wall_seconds=0)])
        (self.workspace / 'state.json').write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, 'previous native child'):
            self.run_task()
        self.assertEqual(len(list((self.workspace / 'runs').iterdir())), 1)

    def test_symlinked_inputs_follow_contents_but_symlink_outputs_are_rejected(self):
        linked = self.root / 'linked'
        linked.mkdir()
        (linked / 'factor.json').symlink_to(self.root / 'factor.json')
        self.spec['stages'][1]['inputs'].append('linked')
        self.save()
        self.run_task()
        (self.root / 'factor.json').write_text('2')
        current = self.run_task()
        self.assertEqual(current['next_action']['skill'], 'precision-repair')
        self.assertEqual([s['reused'] for s in current['stages']], [True, False])
        self.spec['stages'] = [dict(id='link', skill='framework-adapt', timeout=5,
            inputs=['factor.json'], outputs={'result': 'result.json'}, argv=[sys.executable, '-c',
            'from pathlib import Path;import sys;Path(sys.argv[1]).symlink_to(Path("factor.json").resolve())',
            '{run}/result.json'])]
        self.save()
        current = self.run_task()
        self.assertEqual(current['status'], 'needs_action')
        self.assertIn('escaped', current['next_action']['reason'])

    def test_literal_argv_and_output_directory_reuse(self):
        literal = '$(touch injected); `touch injected` $HOME *'
        self.spec['stages'] = [dict(id='literal', skill='precision-compare', timeout=5,
            inputs=[], outputs={'bundle': 'bundle'}, argv=[sys.executable, '-c',
            'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/"text").write_text(sys.argv[2])',
            '{run}/bundle', literal])]
        self.save()
        first = self.run_task()
        folder = Path(first['stages'][0]['outputs']['bundle']['path'])
        self.assertEqual((folder / 'text').read_text(), literal)
        self.assertFalse((self.root / 'injected').exists())
        self.assertTrue(self.run_task()['stages'][0]['reused'])


if __name__ == '__main__':
    unittest.main()
