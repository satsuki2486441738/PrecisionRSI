import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('validate_immutability', ROOT / '.github/validate_immutability.py')
immutability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(immutability)


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

    def test_evidence_may_only_be_added(self):
        changes = ['A\tsite/evidence/new.zip', 'M\tsite/evidence/old.md', 'D\tsite/evidence/gone.json']
        self.assertEqual(immutability.compare_evidence(changes), [
            'evidence site/evidence/old.md changed (M)', 'evidence site/evidence/gone.json changed (D)'])

    def test_current_history_matches_its_own_revision(self):
        self.assertEqual(immutability.validate('HEAD'), [])


if __name__ == '__main__':
    unittest.main()
