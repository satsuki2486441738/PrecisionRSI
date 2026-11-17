import importlib.util
import pathlib
import unittest
import json
import subprocess
import tempfile
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
from precision_maintenance import validate_immutability as immutability


class PublishedHistoryTest(unittest.TestCase):
    base = {'schema_version': 1, 'updated_at': '2026-11-07', 'evaluations': [
        {'id': 'a', 'date': '2026-11-01', 'metrics': [1.5]},
        {'id': 'b', 'date': '2026-11-02', 'evidence': ['evidence/b.md']},
    ]}

    def check(self, head):
        return immutability.compare_documents('evaluations.json', self.base, head, ('evaluations',))

    def test_appending_records_and_refreshing_update_time_is_allowed(self):
        head = {**self.base, 'updated_at': '2026-11-14',
                'evaluations': [*self.base['evaluations'], {'id': 'c', 'date': '2026-11-14'}]}
        self.assertEqual(self.check(head), [])

    def test_rewritten_metric_date_or_evidence_is_rejected(self):
        for field, value in (('metrics', [1.4]), ('date', '2026-11-03'), ('evidence', ['evidence/c.md'])):
            records = [dict(record) for record in self.base['evaluations']]
            target = records[0] if field != 'evidence' else records[1]
            target[field] = value
            with self.subTest(field=field):
                self.assertEqual(len(self.check({**self.base, 'evaluations': records})), 1)

    def test_removed_record_and_schema_change_are_rejected(self):
        head = {**self.base, 'schema_version': 2, 'evaluations': self.base['evaluations'][1:]}
        self.assertEqual(self.check(head), ['evaluations.json: schema_version changed',
                                            'evaluations.json: evaluations record a was removed'])

    def test_nested_json_types_cannot_be_rewritten(self):
        for before, after in [(False, 0), (True, 1), (1, 1.0)]:
            with self.subTest(before=before, after=after):
                base = {'schema_version': 1, 'evaluations': [
                    {'id': 'a', 'settings': {'nested': [before]}}]}
                head = {'schema_version': 1, 'evaluations': [
                    {'id': 'a', 'settings': {'nested': [after]}}]}
                self.assertEqual(immutability.compare_documents(
                    'evaluations.json', base, head, ('evaluations',)),
                    ['evaluations.json: evaluations record a was modified'])

    def test_unidentified_records_and_schema_keep_json_types(self):
        base = {'schema_version': 1, 'evaluations': [{'value': False}]}
        head = {'schema_version': True, 'evaluations': [{'value': 0}]}
        self.assertEqual(immutability.compare_documents(
            'evaluations.json', base, head, ('evaluations',)), [
                'evaluations.json: schema_version changed',
                'evaluations.json: evaluations[0] without an id changed'])

    def test_object_key_order_is_not_a_history_change(self):
        base = {'schema_version': 1, 'evaluations': [{'id': 'a', 'value': 0}]}
        head = {'evaluations': [{'value': 0, 'id': 'a'}], 'schema_version': 1}
        self.assertEqual(immutability.compare_documents(
            'evaluations.json', base, head, ('evaluations',)), [])

    def test_evidence_may_only_be_added(self):
        for prefix in ('site/evidence/', 'community/site/evidence/'):
            with self.subTest(prefix=prefix):
                changes = [f'A\t{prefix}new.zip', f'M\t{prefix}old.md', f'D\t{prefix}gone.json']
                self.assertEqual(immutability.compare_evidence(changes), [
                    f'evidence {prefix}old.md changed (M)', f'evidence {prefix}gone.json changed (D)'])

    def test_current_history_matches_its_own_revision(self):
        self.assertEqual(immutability.validate('HEAD'), [])

    def migrated_fixture(self, directory, prefix):
        root = pathlib.Path(directory)
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE)
        git('init', '--quiet')
        old = root / prefix
        (old / 'data').mkdir(parents=True)
        (old / 'evidence').mkdir()
        data = {'schema_version': 1, 'evaluations': [{'id': 'frozen', 'value': 1.5}]}
        (old / 'data/evaluations.json').write_text(json.dumps(data))
        (old / 'evidence/proof.json').write_bytes(b'{"value":1.5}\n')
        git('add', '.')
        git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
            'commit', '--quiet', '-m', 'published')
        if prefix == 'site':
            (root / 'community').mkdir()
            old.rename(root / 'community/site')
        return root

    def test_site_migration_preserves_history_but_still_rejects_mutations(self):
        for prefix in ('site', 'community/site'):
            with self.subTest(base_prefix=prefix), tempfile.TemporaryDirectory() as tmp:
                root = self.migrated_fixture(tmp, prefix)
                with patch.object(immutability, 'ROOT', root):
                    self.assertEqual(immutability.validate('HEAD'), [])
                    proof = root / 'community/site/evidence/proof.json'
                    original = proof.read_bytes()
                    proof.write_bytes(b'{"value":0}\n')
                    self.assertTrue(immutability.validate('HEAD'))
                    proof.write_bytes(original)
                    data = root / 'community/site/data/evaluations.json'
                    document = json.loads(data.read_text())
                    document['evaluations'][0]['value'] = 0
                    data.write_text(json.dumps(document))
                    self.assertTrue(immutability.validate('HEAD'))

    def test_site_migration_rejects_missing_evidence_or_catalog(self):
        for missing in ('evidence/proof.json', 'data/evaluations.json'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                root = self.migrated_fixture(tmp, 'site')
                (root / 'community/site' / missing).unlink()
                with patch.object(immutability, 'ROOT', root):
                    self.assertTrue(immutability.validate('HEAD'))

    def test_ci_compares_draft_pushes_with_published_main(self):
        event = {'GITHUB_REF': 'refs/heads/codex/evidence-integration',
                 'BEFORE': 'draft-snapshot'}
        self.assertEqual(immutability.ci_base(event), 'origin/main')

    def test_ci_uses_pr_target_or_previous_published_commit(self):
        self.assertEqual(immutability.ci_base({'BASE_REF': 'release'}), 'origin/release')
        with patch.object(immutability, '_git', return_value='') as git:
            self.assertEqual(immutability.ci_base(
                {'GITHUB_REF': 'refs/heads/main', 'BEFORE': 'published-before'}),
                'published-before')
            git.assert_called_once_with('cat-file', '-e', 'published-before^{commit}')

    def test_ci_handles_initial_and_unavailable_main_predecessors(self):
        self.assertEqual(immutability.ci_base(
            {'GITHUB_REF': 'refs/heads/main', 'BEFORE': '0' * 40}), 'origin/main')
        with patch.object(immutability, '_git', side_effect=
                          immutability.subprocess.CalledProcessError(128, ['git'])):
            self.assertEqual(immutability.ci_base(
                {'GITHUB_REF': 'refs/heads/main', 'BEFORE': 'unavailable'}), 'origin/main')


if __name__ == '__main__':
    unittest.main()
