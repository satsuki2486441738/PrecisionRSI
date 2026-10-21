"""Regressions for published repair records and evidence boundaries."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('validate_history', ROOT / '.github/validate_history.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RepairHistoryTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / 'site/data/repairs.json').read_text(encoding='utf-8'))

    def check_invalid(self, mutate):
        with tempfile.TemporaryDirectory() as temp:
            site = Path(temp)
            (site / 'data').mkdir()
            data = copy.deepcopy(self.data)
            # Supply all valid local evidence before mutating the record.
            for repair in data['repairs']:
                for item in repair['evidence']:
                    path = site / item['url']
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('evidence')
            mutate(data)
            (site / 'data/repairs.json').write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                MODULE.validate(site)

    def test_published_history(self):
        self.assertGreaterEqual(MODULE.validate(ROOT / 'site'), 1)

    def test_duplicate_identity(self):
        self.check_invalid(lambda d: d['repairs'].append(copy.deepcopy(d['repairs'][0])))

    def test_invalid_coverage(self):
        self.check_invalid(lambda d: d['repairs'][0]['validation']['coverage'].update(observed=33, expected=32))

    def test_nonfinite_metric(self):
        self.check_invalid(lambda d: d['repairs'][0]['validation']['metrics'][0].update(after=float('nan')))

    def test_escaped_evidence(self):
        self.check_invalid(lambda d: d['repairs'][0]['evidence'][0].update(url='../README.md'))

    def test_unsafe_evidence_scheme(self):
        self.check_invalid(lambda d: d['repairs'][0]['evidence'][0].update(url='javascript:alert(1)'))
