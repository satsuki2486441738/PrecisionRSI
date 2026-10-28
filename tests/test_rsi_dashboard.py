"""Public RSI snapshots retain costs and outcomes without exporting private traces."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rsi_snapshot', ROOT / '.github/rsi_snapshot.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DashboardTests(unittest.TestCase):
    def write(self, root, name, value):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))

    def fixture(self, root, reused=False):
        self.write(root, 'status.json', dict(status='completed', model='test', spent_seconds=9, budget_seconds=60,
            baseline=dict(correct=True, reused=reused, seconds=4, usage=dict(input_tokens=7, output_tokens=3, total_tokens=10)),
            evolution=dict(accepted=0, active_version='empty', reason='no measured progress')))
        return root

    def test_reused_baseline_cost_is_not_new_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = module.summarize(self.fixture(Path(tmp), reused=True))
            self.assertEqual(r['phases']['baseline']['seconds'], 0)
            self.assertEqual(r['phases']['baseline']['usage']['total_tokens'], 0)
            self.assertEqual(r['baseline']['historical']['usage']['total_tokens'], 10)
            self.assertTrue(r['baseline']['correct'])

    def test_unknown_usage_is_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = module.summarize(self.fixture(Path(tmp)))
            self.assertIsNone(r['phases']['evaluation']['usage']['total_tokens'])
            self.assertIsNone(r['phases']['construction']['seconds'])
            self.assertEqual(r['phases']['baseline']['usage']['input_tokens'], 7)

    def test_native_rejection_and_private_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp))
            native = dict(version='a' * 64, construction=dict(seconds=2, usage=dict(total_tokens=6)))
            feedback = dict(errors=[dict(reason='bad /home/private/capture.json token=SECRET')],
                updates=[dict(kind='tool', content='Compare /home/private/tool.py', artifacts=['/home/private/compare.py'])],
                measurements=dict(candidate=dict(correct=False, total_tokens=12)))
            self.write(root, 'evolution/evolution.json', dict(rounds=[dict(workspace='/home/private/round1', result=native, feedback=feedback, accepted=False)]))
            r = module.summarize(root)
            self.assertEqual(r['rounds'][0]['status'], 'rejected')
            self.assertFalse(r['rounds'][0]['correct'])
            self.assertEqual(r['rounds'][0]['changes'][0]['artifacts'], ['compare.py'])
            self.assertEqual(r['rounds'][0]['candidate_tokens'], 12)
            for secret in ('/home/private', 'SECRET', 'workspace', 'stderr'):
                self.assertNotIn(secret, json.dumps(r))

    def test_activity_accepts_string_and_object_tool_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp))
            self.write(root, 'evolution/evolution.json', dict(rounds=[dict(workspace='/private/round1', result={}, feedback={}, accepted=False)]))
            p = root / 'evolution/round1/session/agent/runs/call1/stdout.bin'
            p.parent.mkdir(parents=True)
            p.write_text('\n'.join(json.dumps(e) for e in [dict(type='tool_call', tool='bash'), dict(type='tool_call', tool=dict(name='read')), dict(type='thinking', text='SECRET')]))
            r = module.summarize(root)
            self.assertEqual(r['rounds'][0]['activity']['tools'], dict(bash=1, read=1))
            self.assertNotIn('SECRET', json.dumps(r))

    def test_duplicate_sources_export_once_and_keep_failed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.fixture(root / 'runs/compare-rsi-20261028T000000Z')
            (run / 'REPORT.md').write_text('private report')
            status = json.loads((run / 'status.json').read_text());status['status'] = 'failed'
            self.write(run, 'status.json', status)
            monitor = dict(latest=dict(local=str(run), remote='/private'), previous=dict(local=str(run)), history=[])
            self.write(root, 'monitor.json', monitor)
            packet = module.export(root / 'monitor.json', root / 'rsi.json')
            self.assertEqual(len(packet['runs']), 1)
            self.assertEqual(packet['runs'][0]['status'], 'failed')
            self.assertIsNone(packet['observed_at'])
            self.assertNotIn(str(root), json.dumps(packet))

    def test_no_stage_evaluation_does_not_claim_a_candidate_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp))
            self.write(root, 'evolution/evolution.json', dict(rounds=[dict(result={}, feedback=dict(errors=[dict(reason='missing proposals')]), accepted=False)]))
            r = module.summarize(root)['rounds'][0]
            self.assertFalse(r['status'] == 'accepted')
            self.assertIsNone(r['correct'])
            self.assertIsNone(r['candidate_tokens'])


if __name__ == '__main__':
    unittest.main()
