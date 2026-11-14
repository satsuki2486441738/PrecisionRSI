import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('integration', ROOT / '.github/plan_history_integration.py')
integration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(integration)
EXPORT_SPEC = importlib.util.spec_from_file_location('statistics', ROOT / '.github/export_statistics.py')
statistics = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(statistics)


class IntegrationPlanTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record = {'id': 'original', 'date': '2026-11-01', 'status': 'partial',
                       'acceptance': None, 'comparisons': [{'metrics': [{'value': 0.125}]}]}

    def source(self, name, records):
        path = self.root / name / 'data/evaluations.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'schema_version': 1, 'updated_at': '2026-11-01',
                                    'evaluations': records}), encoding='utf-8')
        return path

    def test_union_preserves_base_metadata_records_order_and_all_sources(self):
        base = self.source('base', [self.record])
        extra = {**self.record, 'id': 'extra', 'evidence': []}
        source = self.source('local', [self.record, extra])
        before = base.read_bytes(), source.read_bytes()
        candidate, audit = integration.plan(base, [('local', source)])
        self.assertEqual(candidate, {'schema_version': 1, 'updated_at': '2026-11-01',
                                     'evaluations': [self.record, extra]})
        self.assertEqual(audit['memberships']['original'], ['base', 'local'])
        self.assertEqual(audit['added_records'], 1)
        self.assertEqual(before, (base.read_bytes(), source.read_bytes()))
        self.assertEqual(integration.plan(base, [('local', source)]), (candidate, audit))

    def test_changed_metric_date_status_acceptance_or_evidence_is_a_conflict(self):
        base = self.source('base', [self.record])
        for key, value in [('date', '2026-11-02'), ('status', 'completed'),
                           ('acceptance', True), ('comparisons', []), ('evidence', [])]:
            with self.subTest(key=key):
                other = {**self.record, key: value}
                candidate, audit = integration.plan(base, [('local', self.source(key, [other]))])
                self.assertIsNone(candidate)
                self.assertEqual(audit['conflicts'][0]['id'], 'original')

    def test_duplicate_ids_keys_and_nonfinite_numbers_rejected(self):
        path = self.source('duplicate', [self.record, self.record])
        with self.assertRaisesRegex(ValueError, 'duplicate evaluation'):
            integration.load(path)
        for text in ['{"evaluations": [], "evaluations": []}',
                     '{"evaluations": [{"id":"x", "value":NaN}]}',
                     '{"evaluations": [{"id":"x", "value":1e999}]}']:
            path.write_text(text)
            with self.assertRaises(ValueError):
                integration.load(path)

    def test_conflicts_between_two_new_sources_are_not_hidden(self):
        base = self.source('base', [])
        one = self.source('one', [self.record])
        two = self.source('two', [{**self.record, 'acceptance': True}])
        candidate, audit = integration.plan(base, [('one', one), ('two', two)])
        self.assertIsNone(candidate)
        self.assertEqual(audit['conflicts'][0]['first_source'], 'one')

    def test_numeric_boolean_type_changes_are_conflicts(self):
        base = self.source('base', [dict(self.record, acceptance=1)])
        candidate, _ = integration.plan(base, [('other', self.source('other', [
            dict(self.record, acceptance=True)]))])
        self.assertIsNone(candidate)

    def test_each_source_uses_own_evidence_root_and_reports_privacy_limits(self):
        base = self.source('base', [])
        record = {**self.record, 'archive': {'published': False}, 'hardware': 'H20Z',
                  'evidence': [{'url': 'evidence/result.json'}, {'url': 'evidence/raw.zip'},
                               {'url': 'evidence/missing.json'}]}
        path = self.source('other', [record])
        folder = path.parent.parent / 'evidence'
        folder.mkdir()
        (folder / 'result.json').write_text('{"PodUID":"private"}')
        (folder / 'raw.zip').write_bytes(b'archive')
        _, audit = integration.plan(base, [('other', path)])
        added = audit['additions'][0]
        self.assertIn('private_archive_provenance', added['findings'])
        self.assertIn('hardware_display_alias_requires_H200', added['findings'])
        self.assertIn('evidence_runtime_identifier_review', added['evidence'][0]['findings'])
        self.assertIn('raw_or_archive_content_not_reviewed', added['evidence'][1]['findings'])
        self.assertIn('missing_evidence_file', added['evidence'][2]['findings'])
        self.assertFalse(audit['limits']['privacy_certified'])

    def test_escape_encoded_traversal_symlink_and_external_urls(self):
        path = self.source('other', [])
        site = path.parent.parent
        (self.root / 'secret').write_text('secret')
        (site / 'link').symlink_to(self.root / 'secret')
        record = {'evidence': [{'url': url} for url in (
            '../../secret', '%2e%2e/secret', 'link', 'file:///secret',
            '//example.test/file', 'https://example.test/file')]}
        entries = integration.evidence_findings(record, site)['evidence']
        self.assertTrue(all(x['findings'] == ['unsafe_evidence_path'] for x in entries[:5]))
        self.assertEqual(entries[-1]['findings'], ['external_evidence_unverified'])
        self.assertTrue(all('sha256' not in x for x in entries))

    def test_size_limit_and_scan_limit_are_explicit(self):
        path = self.source('other', [])
        evidence = path.parent.parent / 'large.json'
        with evidence.open('wb') as stream:
            stream.truncate(integration.MAX_PUBLIC_BYTES + 1)
        entries = integration.evidence_findings({'evidence': [{'url': 'large.json'}]},
                                                path.parent.parent)['evidence']
        self.assertIn('exceeds_public_size_limit', entries[0]['findings'])
        self.assertIn('text_content_exceeds_scan_limit', entries[0]['findings'])

    def test_schema_and_duplicate_source_names_rejected(self):
        base = self.source('base', [])
        with self.assertRaisesRegex(ValueError, 'reserved source'):
            integration.plan(base, [('base', base)])
        other = self.source('other', [])
        data = json.loads(other.read_text())
        data['schema_version'] = 2
        other.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'schema_version'):
            integration.plan(base, [('other', other)])

    def test_statistics_keep_numbers_status_scope_and_unknown_acceptance(self):
        base = self.source('base', [])
        record = {**self.record, 'summary': 'H20Z bounded component only',
                  'whole_task_completed': False,
                  'notes': ['Not a backbone training workload.'],
                  'evidence': [{'url': 'private.zip'}],
                  'archive': {'local_ref': 'refs/private', 'replay_cases': 4,
                              'replay_cases_scope': 'Dependent component cases only'}}
        source = self.source('source', [record])
        result = statistics.export(base, source)
        output = result['evaluations'][0]
        self.assertEqual(output['comparisons'], record['comparisons'])
        self.assertEqual(output['status'], 'partial')
        self.assertIsNone(output['acceptance'])
        self.assertFalse(output['whole_task_completed'])
        self.assertEqual(output['notes'], record['notes'])
        self.assertEqual(output['summary'], 'H200 bounded component only')
        self.assertEqual(output['provenance']['retained_source_scope'], 'Dependent component cases only')
        self.assertFalse(output['provenance']['independent_replay_provided'])
        self.assertNotIn('archive', output)
        self.assertNotIn('evidence', output)
        self.assertNotIn('refs/private', json.dumps(result))
        self.assertEqual(statistics.export(base, source), result)

    def test_statistics_reject_unknown_fields_private_details_and_conflicts(self):
        base = self.source('base', [])
        for record in [dict(self.record, raw_responses=['private']),
                       dict(self.record, notes=['ssh lion']),
                       dict(self.record, notes=['PodUID is private'])]:
            with self.assertRaises(ValueError):
                statistics.export(base, self.source('source', [record]))
        base = self.source('base', [self.record])
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            statistics.export(base, self.source('source', [dict(self.record, acceptance=True)]))

    def test_statistics_skip_exact_existing_records(self):
        base = self.source('base', [self.record])
        result = statistics.export(base, self.source('source', [self.record]))
        self.assertEqual(result['evaluations'], [])


if __name__ == '__main__':
    unittest.main()
