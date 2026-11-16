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
        changes = ['A\tsite/evidence/new.zip', 'M\tsite/evidence/old.md', 'D\tsite/evidence/gone.json']
        self.assertEqual(immutability.compare_evidence(changes), [
            'evidence site/evidence/old.md changed (M)', 'evidence site/evidence/gone.json changed (D)'])

    def test_current_history_matches_its_own_revision(self):
        self.assertEqual(immutability.validate('HEAD'), [])


if __name__ == '__main__':
    unittest.main()
