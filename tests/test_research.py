"""Durable memory, idempotent turns and native-session adapter regressions."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

from precision_rsi.memory import read, write
from precision_rsi.research import Research, document, history, save_handoff, state
from precision_rsi.snapshots import Journal, seal


ROOT = Path(__file__).resolve().parents[1]


class ResearchTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.research = Research(self.root/'research', {'task':'fixed'})
        self.rounds = [dict(result=dict(status='completed', candidates=[]),
            feedback=dict(construction=dict(seconds=1,usage=None), measurements={}),
            accepted=False, repeated=False) for _ in range(9)]
        self.research.sync(self.rounds)

    def test_full_history_pages_and_does_not_expose_host_payload(self):
        self.rounds[0]['feedback']['workflow'] = {'private':'hidden-host-evaluator'}
        self.rounds[0]['feedback']['measurements'] = {'baseline':{'correct':True, 'path':'private-path'}}
        self.rounds[0]['result']['reason'] = 'private-credential'
        self.research.sync(self.rounds)
        first = history(self.research.view)
        second = history(self.research.view, first['next_start'])
        self.assertEqual(first['total'], 9)
        self.assertEqual(second['records'][-1]['round'], 9)
        self.assertNotIn('private', json.dumps(first))
        self.assertEqual(first['records'][0]['measurements']['baseline'], {'correct':True})
        self.assertIsNone(first['records'][0]['construction']['usage'])
        with self.assertRaises(ValueError):
            history(self.research.view, count=6)

    def test_notes_references_are_validated_and_rejoin_authoritative_outcomes(self):
        note = dict(hypotheses=[dict(claim='try streaming', evidence_rounds=[1,9])])
        result = document(self.research.view, 'notes', note)
        self.assertEqual([v['round'] for v in result['evidence']], [1,9])
        for invalid in (dict(observations=['x'*4096]),
                        dict(hypotheses=[dict(claim='unsupported',evidence_rounds=[10])]),
                        dict(hypotheses=[dict(claim='unsupported',evidence_rounds=[True])])):
            with self.assertRaises(ValueError):
                document(self.research.view, 'notes', invalid)
        self.assertEqual(document(self.research.view, 'notes')['judgments'], note)
        document(self.research.view, 'plan', dict(next_experiments=[dict(
            intervention='stream reads', falsified_if='same cost', evidence_rounds=[2])]))

    def test_ledger_retains_execution_and_accounted_cost_separately(self):
        candidate=dict(correct=False,
            execution=dict(seconds=480,total_tokens=500,usage=dict(cached_input_tokens=None)),
            including_construction=dict(seconds=1100,total_tokens=900,cost=None))
        self.rounds[0]['feedback']['measurements']=dict(candidate=candidate,amortize_over_tasks=2)
        self.research.sync(self.rounds)
        measurements=history(self.research.view)['records'][0]['measurements']
        self.assertEqual(measurements['candidate'],candidate)
        self.assertEqual(measurements['amortize_over_tasks'],2)

    def test_filtered_recent_history_finds_old_candidates_and_keeps_round_ids(self):
        candidate = 'a'*64
        for n, row in enumerate(self.rounds):
            row['accepted'] = n % 2 == 0
            row['result']['candidates'] = [candidate] if n in (0, 6, 8) else []
        self.rounds[6]['result']['status'] = 'needs_action'
        self.research.sync(self.rounds)
        first = history(self.research.view, newest=True, outcome='accepted',
                        candidate=candidate, count=2)
        self.assertEqual((first['total'], first['ledger_total']), (3, 9))
        self.assertEqual([r['round'] for r in first['records']], [9, 7])
        second = history(self.research.view, first['next_start'], newest=True,
                         outcome='accepted', candidate=candidate, count=2)
        self.assertEqual([r['round'] for r in second['records']], [1])
        self.assertIsNone(second['next_start'])
        self.assertEqual(history(self.research.view, outcome='incomplete')['records'][0]['round'], 7)
        self.assertEqual(history(self.research.view, outcome='unaccepted')['total'], 4)
        self.assertEqual(history(self.research.view, candidate='b'*64)['records'], [])
        self.assertEqual(history(self.research.view, start=99)['records'], [])
        for options in ({'outcome':'failed'}, {'newest':1}, {'candidate':'../../private'}):
            with self.assertRaises(ValueError):
                history(self.research.view, **options)

    def test_state_is_bounded_read_only_and_preserves_judgment_provenance(self):
        fresh = Research(self.root/'fresh-research', {'task':'new'})
        self.assertEqual(fresh.context(sys.executable)['summary']['rounds'], 0)
        self.assertFalse((fresh.view/'ledger.json').exists())
        self.rounds[2]['accepted'] = True
        self.rounds[-1]['result']['status'] = 'needs_action'
        self.research.sync(self.rounds)
        document(self.research.view, 'notes', dict(hypotheses=[dict(
            claim='unverified hypothesis', evidence_rounds=[3])]))
        document(self.research.view, 'plan', dict(next_experiments=[dict(
            intervention='test hypothesis', falsified_if='no measured gain', evidence_rounds=[9])]))
        before = {p.name:p.read_bytes() for p in self.research.view.glob('*.json')}
        snapshot = state(self.research.view)
        self.assertEqual(snapshot['summary'], dict(rounds=9, accepted_rounds=1,
            incomplete_rounds=1, last_accepted_round=3))
        self.assertEqual(snapshot['documents']['notes']['hypotheses'][0]['evidence_rounds'], [3])
        self.assertNotIn('records', snapshot)
        self.assertNotIn('private', json.dumps(snapshot))
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.research.view.glob('*.json')})
        result = subprocess.run([sys.executable, '-B', '-m', 'precision_rsi.research',
            str(self.research.view), 'history', '--outcome', 'incomplete', '--newest'],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['records'][0]['round'], 9)
        self.research.sync([])
        self.assertEqual(state(self.research.view)['summary']['last_accepted_round'], None)

    def test_current_judgments_reattach_without_full_ledger_or_trace(self):
        document(self.research.view, 'notes', dict(observations=['try a different parser']))
        context = self.research.context(sys.executable)
        self.assertEqual(context['documents']['notes']['observations'], ['try a different parser'])
        self.assertNotIn('private', json.dumps(context))
        self.assertNotIn('ledger.json', json.dumps(context))
        (self.research.view/'notes.json').write_text('invalid JSON')
        self.assertIn('error', self.research.context(sys.executable)['documents']['notes'])

    def test_declared_schemas_are_valid_before_any_observed_round(self):
        self.research.sync([])
        context = self.research.context(sys.executable)
        for kind, example in context['schemas'].items():
            self.assertEqual(document(self.research.view, kind, example)['judgments'], example)
        self.assertIn('--set replaces', context['semantics'])

    def test_result_handoff_validates_both_documents_before_writing(self):
        notes = dict(observations=['retain the current hypothesis'])
        document(self.research.view, 'notes', notes)
        before = (self.research.view/'notes.json').read_bytes()
        with self.assertRaises(ValueError):
            save_handoff(self.research.view, dict(notes=dict(observations=['changed']),
                plan=dict(next_experiments=[dict(intervention='try parser',
                    falsified_if='no gain', evidence_rounds=[10])])))
        self.assertEqual((self.research.view/'notes.json').read_bytes(), before)
        self.assertFalse((self.research.view/'plan.json').exists())
        changes = dict(notes=dict(observations=['changed']), plan=dict(next_experiments=[]))
        save_handoff(self.research.view, changes)
        save_handoff(self.research.view, changes)
        self.assertEqual(state(self.research.view)['documents'], changes)
        with self.assertRaises(ValueError):
            save_handoff(self.research.view, {'accepted':True})
        with self.assertRaises(ValueError):
            save_handoff(self.research.view, {'notes':None})
        self.assertEqual(state(self.research.view)['documents'], changes)

    def test_controller_checkpoint_survives_without_model_notes_and_exports(self):
        run = self.root/'run'
        artifacts = run/'artifacts'
        artifacts.mkdir(parents=True)
        (artifacts/'candidate.py').write_text('print("unverified private content")')
        observed = dict(input_tokens=8, output_tokens=2, total_tokens=10, steps=1,
                        cached_input_tokens=None)
        self.research.checkpoint('round-001', run, status='interrupted', usage=observed)
        current = self.research.context(sys.executable)['handoff']
        self.assertEqual(current['artifacts'], ['candidate.py'])
        self.assertEqual(current['next_action'], 'inspect_retained_work')
        self.assertEqual(current['reported_usage']['total_tokens'], 10)
        self.assertIsNone(current['reported_usage']['cached_input_tokens'])
        self.assertNotIn(str(run), json.dumps(current))
        self.assertNotIn('private content', json.dumps(current))
        self.assertEqual(state(self.research.view)['documents'], {})
        store = self.root/'memory'; store.mkdir()
        journal = Journal(self.root/'snapshots')
        event = seal(self.root, store, 'handoff', 'committed', {},
                     journal=journal, research=self.research)
        journal.restore(event['id'], self.root/'restored')
        self.assertEqual(read(self.root/'restored/session/research/handoff.json'), current)

    @unittest.skipUnless(os.name == 'posix', 'process-group interruption requires POSIX')
    def test_completed_step_checkpoint_survives_hard_kill_without_final_reply(self):
        script = self.root/'partial.py'
        script.write_text('''import json,time
from pathlib import Path
(Path('artifacts')/'partial.py').write_text('partial implementation')
print(json.dumps({'type':'status','phase':'step_end',
 'usage':{'inputTokens':10,'outputTokens':2,'totalTokens':12}}),flush=True)
time.sleep(30)
''')
        run = self.root/'partial-run'; run.mkdir()
        cfg = self.root/'agent.json'
        write(cfg, dict(argv=[sys.executable, str(script), '{prompt}'], usage_format='dsh-jsonl',
            research=dict(root=str(self.research.root), binding={'task':'fixed'}, turn='round-001')))
        prompt = self.root/'prompt.txt'; prompt.write_text('partial task')
        with subprocess.Popen([sys.executable, '-B', '-m', 'precision_rsi.harness',
                str(cfg), str(prompt), str(run), '--cwd-run'], cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True) as process:
            try:
                import selectors
                with selectors.DefaultSelector() as ready:
                    ready.register(process.stdout, selectors.EVENT_READ)
                    self.assertTrue(ready.select(timeout=5), 'no completed step received')
                event = json.loads(process.stdout.readline())
                self.assertEqual(event['phase'], 'step_end')
            finally:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=5)
        current = state(self.research.view)['handoff']
        self.assertEqual(current['status'], 'running')  # No fabricated graceful exit.
        self.assertEqual(current['reported_usage']['total_tokens'], 12)
        self.assertEqual(current['artifacts'], ['partial.py'])
        self.assertTrue((self.research.private/'active.json').exists())
        self.assertFalse((self.research.private/'turn-round-001.json').exists())

    def test_identity_turn_receipts_and_explicit_interruption(self):
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            Research(self.research.root, {'task':'different'})
        self.assertIsNone(self.research.begin('round-001', {'input':1}))
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            self.research.begin('round-002', {'input':2})
        self.research.complete('round-001', status='completed')
        # Receipt committed, active unlink interrupted.
        write(self.research.private/'active.json', dict(turn='round-001', key='old'))
        self.assertEqual(self.research.begin('round-001', {'input':1})['status'], 'completed')
        self.assertFalse((self.research.private/'active.json').exists())
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            self.research.begin('round-001', {'input':2})
        self.research.begin('round-002', {'input':2})
        self.research.reconcile('round-002', 'operator confirms process stopped')
        self.assertIsNone(self.research.begin('round-003', {'input':3}))

    def test_snapshots_separate_visible_memory_and_private_native_history(self):
        native = self.research.private/'native/session.jsonl'
        native.parent.mkdir()
        native.write_text('{"reasoning":"private-native-history"}\n')
        document(self.research.view, 'notes', dict(observations=['rejected']))
        store = self.root/'memory'
        store.mkdir()
        journal = Journal(self.root/'snapshots')
        event = seal(self.root, store, 'committed', 'committed', {},
                     research=self.research, journal=journal)
        coverage = read(journal.root/'objects'/event['files']['audit/context-coverage.json']['object'])
        self.assertTrue(coverage['native_session_exported'])
        restored = self.root/'visible'
        journal.restore(event['id'], restored)
        self.assertTrue((restored/'session/research/notes.json').is_file())
        self.assertFalse((restored/'audit').exists())
        operator = self.root/'operator'
        journal.restore(event['id'], operator, view='operator')
        self.assertEqual((operator/'audit/research/native/session.jsonl').read_text(), native.read_text())

    def test_adapter_reuses_native_session_cwd_and_completed_turn(self):
        # Fake DSH implements the documented CLI; real DSH integration is optional.
        script = self.root/'dsh.py'
        script.write_text('''import json,sys
from pathlib import Path
args=sys.argv
assert args.index('--patch') < args.index('--json')
patch=json.loads(Path(args[args.index('--patch')+1]).read_text())
root=Path(patch[0]['config']['root']); root.mkdir(exist_ok=True)
log=root/'session.jsonl'
session=args[args.index('--session-id')+1] if '--session-id' in args else 'native-123'
with log.open('a') as f: f.write(json.dumps({'cwd':str(Path.cwd()),'prompt':args[-1]})+'\\n')
print(json.dumps({'type':'session','sessionId':session}))
print(json.dumps({'type':'status','phase':'step_end','usage':{'inputTokens':5,'outputTokens':2,'totalTokens':7}}))
''')
        agent = dict(argv=[sys.executable, str(script), '--json', '{prompt}'],
                     usage_format='dsh-jsonl', persistence={'kind':'dsh'})

        def invoke(number):
            run = self.root/f'run{number}'
            run.mkdir(exist_ok=True)
            info = dict(root=str(self.research.root), binding={'task':'fixed'}, turn=f'round-{number:03d}')
            cfg = self.root/'agent.json'
            write(cfg, dict(agent, research=info))
            prompt = self.root/'prompt.txt'
            prompt.write_text(f'round {number}')
            return subprocess.run([sys.executable, '-B', '-m', 'precision_rsi.harness',
                str(cfg), str(prompt), str(run), '--cwd-run'], cwd=ROOT, capture_output=True)

        first = invoke(1)
        self.assertEqual(first.returncode, 0, first.stderr.decode())
        self.assertEqual(invoke(1).stdout, first.stdout)
        second = invoke(2)
        self.assertEqual(second.returncode, 0, second.stderr.decode())
        logs = [json.loads(v) for v in (self.research.private/'native/session.jsonl').read_text().splitlines()]
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[0]['cwd'], logs[1]['cwd'])
        self.assertEqual(read(self.research.private/'cursor.json')['session_id'], 'native-123')
        receipt = read(self.research.private/'turn-round-002.json')
        self.assertEqual(receipt['usage']['total_tokens'], 7)
        (self.research.private/'native/session.jsonl').unlink()
        third = invoke(3)
        self.assertNotEqual(third.returncode, 0)
        self.assertIn(b'explicit recovery required', third.stderr)
        fourth = invoke(4)
        self.assertIn(b'unfinished researcher turn', fourth.stderr)


if __name__ == '__main__':
    unittest.main()
