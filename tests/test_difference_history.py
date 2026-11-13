"""Survey provenance rejects corrupted measurements without rounding tiny values."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('difference_validator', ROOT / '.github/validate_differences.py')
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class DifferenceHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / 'site/data/differences.json').read_text(encoding='utf-8'))
        cls.raw = json.loads((ROOT / 'site/data/differences-raw.json').read_text(encoding='utf-8'))

    def test_all_published_metrics_match_frozen_source(self):
        records, metrics = VALIDATOR.validate_data(self.data, self.raw)
        self.assertEqual(records, len(self.data['records']))
        self.assertGreater(metrics, records)
        self.assertEqual(len(self.data['branches']), 5)
        self.assertGreaterEqual(len(self.data['models']), 7)

    def test_corrupt_measurement_is_rejected(self):
        data = copy.deepcopy(self.data)
        metric = next(m for r in data['records'] for m in r['metrics'] if 'source_pointer' in m)
        metric['value'] += 1
        with self.assertRaisesRegex(ValueError, 'differs from original source'):
            VALIDATOR.validate_data(data, self.raw)

    def test_unresolvable_evidence_and_mutable_revision_are_rejected(self):
        for change in ({'json_pointer': '/missing/evidence'}, {'commit': 'main'}):
            data = copy.deepcopy(self.data)
            data['records'][0]['source'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                VALIDATOR.validate_data(data, self.raw)

    def test_duplicate_measurements_and_nonfinite_values_are_rejected(self):
        data = copy.deepcopy(self.data)
        data['records'].append(copy.deepcopy(data['records'][0]))
        with self.assertRaisesRegex(ValueError, 'distinct stable slugs'):
            VALIDATOR.validate_data(data, self.raw)
        for value in (float('inf'), float('nan'), True):
            data = copy.deepcopy(self.data)
            data['records'][0]['metrics'][0]['value'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'nonfinite metric'):
                VALIDATOR.validate_data(data, self.raw)

    def test_completion_does_not_grant_numerical_acceptance(self):
        data = copy.deepcopy(self.data)
        data['records'][0]['acceptance'] = 'accepted'
        with self.assertRaisesRegex(ValueError, 'must not imply acceptance'):
            VALIDATOR.validate_data(data, self.raw)

    def test_json_pointer_preserves_zero_negative_tiny_and_escaped_keys(self):
        source = {'a/b': {'~key': [0, -0.25, 1e-300]}}
        self.assertEqual(VALIDATOR.resolve_pointer(source, ''), source)
        for index, value in enumerate((0, -0.25, 1e-300)):
            self.assertEqual(VALIDATOR.resolve_pointer(source, f'/a~1b/~0key/{index}'), value)
        with self.assertRaises(ValueError):
            VALIDATOR.resolve_pointer(source, '/a~1b/~0key/99')


if __name__ == '__main__':
    unittest.main()
