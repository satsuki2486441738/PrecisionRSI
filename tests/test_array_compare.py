"""Direct comparisons preserve native equality, coverage and optional arithmetic."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/precision-compare/scripts'))
from array_compare import compare_arrays
from compare_tensors import Stats


class ArrayCompareTests(unittest.TestCase):
    def test_exact_native_values_scalars_empty_and_noncontiguous_arrays(self):
        a = {'loss': np.asarray(1.), 'weights': np.arange(24).reshape(4, 6)[:, ::2],
             'empty': np.zeros((0, 3)), 'mask': np.asarray([True, False])}
        b = {k: v.copy() for k, v in reversed(list(a.items()))}
        report = compare_arrays(a, b, chunk_size=3)
        self.assertTrue(report['exact_equal'])
        self.assertEqual(report['coverage']['compared_elements'], 15)
        self.assertEqual(report['coverage']['compared_arrays'], 4)
        self.assertNotIn('metrics', report)
        json.dumps(report, allow_nan=False)
        np.testing.assert_array_equal(a['weights'], np.arange(24).reshape(4, 6)[:, ::2])

    def test_missing_both_and_unexpected_inventory_are_not_hidden(self):
        a = {'x': np.ones(2), 'extra': np.ones(1)}
        r = compare_arrays(a, {'x': np.ones(2)}, expected=['x', 'shared-missing'])
        self.assertEqual(r['status'], 'not_covered')
        self.assertFalse(r['exact_equal'])
        self.assertEqual(r['coverage']['missing'], {'side_a': 1, 'side_b': 1})
        self.assertEqual(r['coverage']['unexpected'], {'side_a': 1, 'side_b': 0})
        self.assertEqual(r['coverage']['compared_elements'], 2)
        self.assertEqual(len(r['findings']), 3)

    def test_union_inventory_and_bounded_findings_still_count_all_differences(self):
        a = {str(i): np.zeros(7) for i in range(10)}
        b = {str(i): np.ones(7) for i in range(10)}
        b['extra'] = np.ones(2)
        r = compare_arrays(a, b, max_findings=2, chunk_size=3)
        self.assertEqual(r['coverage']['expected_arrays'], 11)
        self.assertEqual(r['coverage']['missing']['side_a'], 1)
        self.assertEqual(r['unequal_elements'], 70)
        self.assertEqual(r['unequal_arrays'], 10)
        self.assertEqual(len(r['findings']), 2)
        self.assertEqual(r['findings_omitted'], 9)

    def test_dtype_and_shape_must_match_even_with_identical_flat_values(self):
        a = {'dtype': np.ones(2, dtype=np.float32), 'shape': np.ones((1, 2)), 'good': np.ones(3)}
        b = {'dtype': np.ones(2, dtype=np.float64), 'shape': np.ones(2), 'good': np.ones(3)}
        r = compare_arrays(a, b, metrics=True)
        self.assertEqual(r['status'], 'incomparable')
        self.assertFalse(r['exact_equal'])
        self.assertEqual(r['coverage']['compared_elements'], 3)
        self.assertEqual(r['incompatible_arrays'], 2)
        self.assertIsNone(r['metrics'])

    def test_nonfinite_values_never_establish_correctness_and_json_stays_finite(self):
        a = {'x': np.array([1., np.inf, np.nan, -np.inf])}
        b = {'x': np.array([2., np.inf, np.nan, 3.])}
        r = compare_arrays(a, b, metrics=True, chunk_size=2)
        self.assertFalse(r['exact_equal'])
        self.assertEqual(r['nonfinite'], {'side_a': 3, 'side_b': 2})
        self.assertEqual(r['unequal_elements'], 3)
        self.assertIsNone(r['metrics'])
        self.assertIn({'name': 'x', 'reason': 'nonfinite', 'index': [1],
                       'values': {'dtype': 'float64', 'side_a': 'inf', 'side_b': 'inf'}}, r['findings'])
        json.dumps(r, allow_nan=False)

    def test_large_integer_mismatch_survives_fp64_rounding(self):
        a = {'counter': np.array([2**60], dtype=np.int64)}
        b = {'counter': np.array([2**60+1], dtype=np.int64)}
        r = compare_arrays(a, b, metrics=True)
        self.assertFalse(r['exact_equal'])
        self.assertEqual(r['unequal_elements'], 1)
        self.assertIsNone(r['metrics'])
        self.assertIn('integer', r['metrics_unavailable'])

    def test_metrics_reuse_existing_reducer_and_preserve_max_coordinate(self):
        a = {'a': np.array(0.), 'z': np.arange(12.).reshape(3, 4)[:, ::2]}
        b = {k: v.copy() for k, v in a.items()}
        b['z'][2, 1] += 7
        expected = Stats()
        for name in sorted(a):
            expected.add(a[name].reshape(-1), b[name].reshape(-1))
        r = compare_arrays(a, b, metrics=True, chunk_size=2)
        for k, v in expected.report().items():
            if isinstance(v, dict) and isinstance(v.get('value'), float):
                self.assertAlmostEqual(r['metrics'][k]['value'], v['value'])
            else:
                self.assertEqual(r['metrics'][k], v)
        self.assertEqual(r['metrics']['max_coordinate'], {'name': 'z', 'index': [2, 1]})
        self.assertEqual(r['findings'][0]['index'], [2, 1])

    def test_explicit_mapping_can_pair_different_layouts_without_manifests(self):
        native_a = {'layer.weight': np.arange(12.).reshape(3, 4)}
        native_b = {'shard_0': np.arange(8.), 'shard_1': np.arange(8., 12.)}
        logical_a = {'step/weight': native_a['layer.weight']}
        logical_b = {'step/weight': np.concatenate([native_b['shard_0'], native_b['shard_1']]).reshape(3, 4)}
        report = compare_arrays(logical_a, logical_b, expected=['step/weight'])
        self.assertTrue(report['exact_equal'])
        self.assertEqual(report['coverage']['compared_elements'], 12)
        native_b['shard_1'][1] = -5
        logical_b['step/weight'] = np.concatenate([native_b['shard_0'], native_b['shard_1']]).reshape(3, 4)
        report = compare_arrays(logical_a, logical_b)
        self.assertEqual(report['findings'], [{'name': 'step/weight', 'reason': 'values', 'index': [2, 1],
            'values': {'dtype': 'float64', 'side_a': '9.0', 'side_b': '-5.0'}}])

    def test_scalar_and_empty_metrics_keep_coordinate_and_undefined_norms(self):
        r = compare_arrays({'x': np.array(1.)}, {'x': np.array(2.)}, metrics=True)
        self.assertEqual(r['metrics']['max_coordinate'], {'name': 'x', 'index': []})
        r = compare_arrays({'empty': np.zeros(0)}, {'empty': np.zeros(0)}, metrics=True)
        self.assertTrue(r['exact_equal'])
        self.assertEqual(r['coverage']['compared_elements'], 0)
        self.assertEqual(r['metrics']['status'], 'not_covered')
        r = compare_arrays({'x': np.zeros(2)}, {'x': np.zeros(2)}, metrics=True)
        self.assertIsNone(r['metrics']['cosine']['value'])
        json.dumps(r, allow_nan=False)

    def test_metric_overflow_does_not_hide_native_inequality(self):
        r = compare_arrays({'x': np.array([1e308])}, {'x': np.array([-1e308])}, metrics=True)
        self.assertFalse(r['exact_equal'])
        self.assertIsNone(r['metrics'])
        self.assertIn('nonfinite', r['metrics_unavailable'])
        json.dumps(r, allow_nan=False)

    def test_numeric_equality_is_not_byte_identity_and_metrics_are_optional(self):
        with patch('array_compare.Stats', side_effect=AssertionError('unrequested metric work')):
            r = compare_arrays({'x': np.array([-0.])}, {'x': np.array([0.])})
        self.assertTrue(r['exact_equal'])

    def test_diagnostic_values_are_native_exact_and_json_safe(self):
        a = {'counter': np.array([2**60], dtype=np.int64),
             'flag': np.array([True]), 'tiny': np.array([np.nan], dtype=np.float32)}
        b = {'counter': np.array([2**60+1], dtype=np.int64),
             'flag': np.array([False]), 'tiny': np.array([np.inf], dtype=np.float32)}
        report = compare_arrays(a, b)
        values = {row['name']: row['values'] for row in report['findings']}
        self.assertEqual(values['counter'], {'dtype': 'int64', 'side_a': str(2**60),
                                             'side_b': str(2**60+1)})
        self.assertEqual(values['flag'], {'dtype': 'bool', 'side_a': 'True', 'side_b': 'False'})
        self.assertEqual(values['tiny'], {'dtype': 'float32', 'side_a': 'nan', 'side_b': 'inf'})
        json.dumps(report, allow_nan=False)

    def test_first_mismatch_and_maximum_error_keep_their_own_values(self):
        a = {'x': np.arange(6., dtype=np.float32).reshape(2, 3)[:, ::2]}
        b = {'x': a['x'].copy()}
        b['x'][0, 0] += 1; b['x'][1, 1] += 7
        report = compare_arrays(a, b, metrics=True, chunk_size=1)
        self.assertEqual(report['findings'][0]['values'],
                         {'dtype': 'float32', 'side_a': '0.0', 'side_b': '1.0'})
        self.assertEqual(report['metrics']['max_coordinate'], {'name': 'x', 'index': [1, 1]})
        self.assertEqual(report['metrics']['max_values'],
                         {'dtype': 'float32', 'side_a': '5.0', 'side_b': '12.0'})
        reverse = compare_arrays(b, a, metrics=True)
        self.assertEqual(reverse['metrics']['max_values']['side_a'], '12.0')
        self.assertEqual(reverse['metrics']['max_values']['side_b'], '5.0')

    def test_invalid_inputs_and_inventories(self):
        for kwargs in ({'expected': ['x', 'x']}, {'expected': 'x'}, {'chunk_size': 0},
                       {'metrics': 1}, {'max_findings': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compare_arrays({'x': np.ones(1)}, {'x': np.ones(1)}, **kwargs)
        for a in ({'x': np.array(['x'])}, {'': np.ones(1)}, {'x': np.array([1j])}, {}):
            with self.subTest(a=a), self.assertRaises(ValueError):
                compare_arrays(a, a)

    def test_cli_npz_reports_difference_without_treating_it_as_execution_failure(self):
        script = Path(__file__).resolve().parents[1] / 'skills/precision-compare/scripts/array_compare.py'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root/'a.npz', x=np.ones(2))
            np.savez(root/'b.npz', x=np.zeros(2))
            argv = [sys.executable, '-I', '-B', str(script), str(root/'a.npz'), str(root/'b.npz'), '--output', str(root/'report.json')]
            p = subprocess.run(argv, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertFalse(json.loads((root/'report.json').read_text())['exact_equal'])
            self.assertEqual(subprocess.run(argv, capture_output=True).returncode, 2)
            (root/'expected.json').write_text('["x", "absent"]')
            argv[-1] = str(root/'incomplete.json')
            p = subprocess.run(argv + ['--expected', str(root/'expected.json')], capture_output=True)
            self.assertEqual(p.returncode, 1, p.stderr)


if __name__ == '__main__':
    unittest.main()
