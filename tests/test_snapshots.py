"""Immutable boundary, isolation and detached recovery regressions."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from precision_rsi import versions
from precision_rsi.memory import fingerprint, propose, read, write
from precision_rsi.snapshots import Journal, seal


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.journal = Journal(self.root/'journal')

    def test_detached_restore_deduplicates_and_preserves_executable(self):
        source = self.root/'tool.py'
        source.write_text('print("original")\n'); source.chmod(0o755)
        first = self.journal.capture('checkpoint-000', 'initial', files={'changes/tool.py': source},
            values={'audit/private.json': {'secret_evaluation': True}, 'session/notes.json': {'goal': 'compare'}})
        second = self.journal.capture('round-001', 'committed', parent=first['id'],
            files={'changes/tool.py': source})
        self.assertEqual(first['files']['changes/tool.py'], second['files']['changes/tool.py'])
        source.unlink()
        output = self.root/'fork'
        self.journal.restore(first['id'], output)
        self.assertEqual((output/'changes/tool.py').read_text(), 'print("original")\n')
        self.assertTrue((output/'changes/tool.py').stat().st_mode & 0o100)
        self.assertFalse((output/'audit').exists())
        self.assertEqual(read(output/'session/notes.json'), {'goal': 'compare'})
        self.assertFalse(read(output/'RESTORE.json')['execution_resumed'])
        self.journal.restore(first['id'], self.root/'operator', view='operator')
        self.assertTrue((self.root/'operator/audit/private.json').exists())
        with self.assertRaisesRegex(ValueError, 'new directory'):
            self.journal.restore(first['id'], output)

    def test_inflight_is_not_resumable_and_replay_does_not_read_live_files(self):
        source = self.root/'candidate'; source.write_text('candidate')
        event = self.journal.capture('round-submitted', 'submitted', files={'changes/tool': source})
        source.unlink()
        self.assertEqual(self.journal.capture('round-submitted', 'submitted'), event)
        with self.assertRaisesRegex(ValueError, 'not a checkpoint'):
            self.journal.restore(event['id'], self.root/'bad')
        self.journal.restore(event['id'], self.root/'evidence', evidence=True)
        self.assertFalse(read(self.root/'evidence/RESTORE.json')['checkpoint'])

    def test_corruption_is_detected_before_export(self):
        event = self.journal.capture('init', 'initial', values={'session/notes': 'hello'})
        blob = self.journal.root/'objects'/event['files']['session/notes']['object']
        blob.write_text('modified')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            self.journal.restore(event['id'], self.root/'output')
        self.assertFalse((self.root/'output').exists())

    def test_paths_and_symlinks_are_rejected(self):
        for name in ('../outside', 'session/../outside', 'session/.env', 'session/harness-state/a', 'audit/id_rsa'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.journal.capture('invalid', 'initial', values={name: 'secret'})
        source = self.root/'file'; source.write_text('x')
        link = self.root/'link'; link.symlink_to(source)
        with self.assertRaisesRegex(ValueError, 'regular'):
            self.journal.capture('link', 'initial', files={'changes/link': link})
        with self.assertRaisesRegex(ValueError, 'collision'):
            self.journal.capture('collision', 'initial', values={'session/a': {}, 'session/a/b': {}})

    def test_event_pointer_and_expected_artifact_are_checked(self):
        event = self.journal.capture('init', 'initial')
        pointer = self.journal.root/'events/init.json'
        pointer.write_text(pointer.read_text().replace('"initial"', '"committed"'))
        with self.assertRaisesRegex(ValueError, 'event'):
            self.journal.event('init')
        source = self.root/'source'; source.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.journal.capture('bad', 'submitted', files={'changes/tool': source},
                                 expected={'changes/tool': '0'*64})
        self.assertFalse((self.journal.root/'events/bad.json').exists())

    def test_resources_are_references_and_rollback_keeps_admission_gate(self):
        resource = self.root/'weights'; resource.write_bytes(b'weights')
        store = self.root/'memory'
        candidate = propose(store, dict(kind='weights', content='model weights', tags=[],
                                       artifacts=[str(resource)]), {})
        event = seal(self.root/'run', store, 'init', 'initial', {}, candidates=[candidate], journal=self.journal)
        self.assertFalse(any(name.endswith('/weights') for name in event['files']))
        self.assertIn('changes/resources.json', event['files'])
        self.assertEqual(self.journal.rollback(event['id'], store)['id'], 'empty')
        with patch('precision_rsi.snapshots._platform', return_value={}):
            with self.assertRaisesRegex(ValueError, 'platform changed'):
                self.journal.rollback(event['id'], store)

    def test_seal_freezes_provenance_and_artifacts_without_original_paths(self):
        tool = self.root/'tool.py'; tool.write_text('value = 42\n')
        store = self.root/'memory'
        candidate = propose(store, dict(kind='tool', content='value', tags=[], artifacts=[str(tool)]), {})
        event = seal(self.root/'run', store, 'submitted', 'submitted', {'version': None},
                     candidates=[candidate], journal=self.journal)
        tool.unlink()
        self.journal.restore(event['id'], self.root/'fork', evidence=True, view='operator')
        bindings = read(self.root/'fork/audit/bindings.json')
        self.assertEqual((self.root/'fork'/bindings[str(tool)]).read_text(), 'value = 42\n')
        self.assertTrue((self.root/'fork/platform/precision_rsi/snapshots.py').is_file())
        self.assertEqual(read(self.root/f'fork/audit/candidates/{candidate}.json')['status'], 'candidate')

    def test_directory_artifact_restores_empty_directories_and_exact_identity(self):
        package = self.root/'package'; (package/'empty').mkdir(parents=True)
        (package/'tool.py').write_text('value = 1\n')
        expected = fingerprint(package)
        store = self.root/'memory'
        candidate = propose(store, dict(kind='tool', content='package.tool', tags=[], artifacts=[str(package)]), {})
        event = seal(self.root/'run', store, 'init', 'initial', {}, candidates=[candidate], journal=self.journal)
        shutil.rmtree(package)
        self.journal.restore(event['id'], self.root/'fork', view='operator')
        binding = read(self.root/'fork/audit/bindings.json')[str(package)]
        self.assertEqual(fingerprint(self.root/'fork'/binding), expected)
        self.assertTrue((self.root/'fork'/binding/'empty').is_dir())

    def test_rollback_rechecks_original_artifacts_and_admission(self):
        source = self.root/'tool.py'; source.write_text('value = 1\n')
        store = self.root/'memory'
        candidate = propose(store, dict(kind='tool', content='tool', tags=[], artifacts=[str(source)]), {})
        version = versions.prepare(store, [candidate])['id']
        (store/'versions/results').mkdir()
        receipt = store/f'versions/results/{version}.json'
        write(receipt, dict(accepted=True, history=[]))
        versions.activate(store, version)
        event = seal(self.root/'run', store, 'accepted', 'committed', {}, journal=self.journal)
        versions.activate(store, 'empty')
        write(receipt, dict(accepted=False, history=[]))
        with self.assertRaisesRegex(ValueError, 'passing evaluation'):
            self.journal.rollback(event['id'], store)
        write(receipt, dict(accepted=True, history=[]))
        self.assertEqual(self.journal.rollback(event['id'], store)['id'], version)
        source.write_text('changed = True\n')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            self.journal.rollback(event['id'], store)


if __name__ == '__main__':
    unittest.main()
