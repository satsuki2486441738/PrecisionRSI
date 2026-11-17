"""Public results retain measurements without promising private downloads."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
from precision_maintenance import prepare_public_results as MODULE


class PublicResultsTests(unittest.TestCase):
    def test_projection_preserves_observations_and_separates_local_evidence(self):
        archives = {'old.zip': dict(name='old.zip', sha256='a'*64, availability='local-only')}
        value = dict(id='retained-id', acceptance=None, metrics=[0, -1.25], coverage=dict(observed=2, expected=3),
                     evidence=[dict(label='metrics', url='data/metrics.json'), dict(label='raw', url='evidence/old.zip')],
                     delivery=dict(download='evidence/old.zip'),
                     assets=[dict(kind='portable_supplement', file='supplements/old.zip', scope='bounded')])
        original = copy.deepcopy(value)
        result = MODULE.project(value, archives)
        self.assertEqual(value, original)
        for key in ('id', 'acceptance', 'metrics', 'coverage'):
            self.assertEqual(result[key], value[key])
        self.assertEqual(result['evidence'], value['evidence'][:1])
        self.assertEqual(result['local_evidence'][0]['sha256'], 'a'*64)
        self.assertNotIn('download', result['delivery'])
        self.assertEqual(result['assets'][0]['kind'], 'local_archive')
        self.assertEqual(MODULE.project(result, archives), result)

    def test_export_hash_describes_projected_bytes(self):
        value = dict(source_sha256='original', export_sha256='old', data=dict(value=0.3, acceptance=None))
        result = MODULE.project(value, {})
        self.assertEqual(result['source_sha256'], 'original')
        self.assertEqual(result['export_sha256'], hashlib.sha256(MODULE.encoded(result['data'])).hexdigest())

    def test_report_retains_label_without_local_archive_link(self):
        text = '[raw](../old.zip) and [metrics](data/result.json)'
        result = MODULE.report_text(text, {'old.zip': {}})
        self.assertIn('local evidence retained', result)
        self.assertNotIn('(../old.zip)', result)
        self.assertIn('[metrics](data/result.json)', result)

    def test_public_snapshot_has_consistent_hashes_and_no_archive_links(self):
        snapshot=json.loads((ROOT/'community/site/data/paper-results.json').read_text())
        self.assertEqual(snapshot['publication']['mode'],'metrics-only')
        for source in snapshot['sources']:
            self.assertEqual(source['export_sha256'],hashlib.sha256(MODULE.encoded(source['data'])).hexdigest())
        def check(value):
            if isinstance(value,dict):
                for key,item in value.items():
                    if key in ('url','download','href') and isinstance(item,str):
                        self.assertNotIn('.zip',item)
                    check(item)
            elif isinstance(value,list):
                for item in value: check(item)
        check(snapshot)
        archives=json.loads((ROOT/'community/site/data/local-evidence.json').read_text())['archives']
        self.assertGreater(len(archives),0)
        self.assertTrue(all(a['availability']=='local-only' and len(a['sha256'])==64 for a in archives))

    def test_results_page_has_no_static_archive_downloads(self):
        import re
        page=(ROOT/'community/site/paper.html').read_text()
        self.assertIsNone(re.search(r'href=["\'][^"\']*\.zip',page))
        self.assertIn('原始证据本地保留，不提供在线下载',page)
