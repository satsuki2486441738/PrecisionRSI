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

    def test_shape_blocks_metrics_while_dtype_differences_remain_observable(self):
        a = {'dtype': np.ones(2, dtype=np.float32), 'shape': np.ones((1, 2)), 'good': np.ones(3)}
        b = {'dtype': np.ones(2, dtype=np.float64), 'shape': np.ones(2), 'good': np.ones(3)}
        r = compare_arrays(a, b, metrics=True)
        self.assertEqual(r['status'], 'incomparable')
        self.assertFalse(r['exact_equal'])
        self.assertEqual(r['coverage']['compared_elements'], 5)
        self.assertEqual(r['incompatible_arrays'], 1)
        self.assertEqual(r['dtype_differences'], 1)
        self.assertIsNone(r['metrics'])

    def test_cross_dtype_metrics_preserve_native_identity_and_error_values(self):
        a = {'x': np.array([1., 1.001], dtype=np.float32)}
        b = {'x': np.array([1., 1.001], dtype=np.float16)}
        for left, right in ((a, b), (b, a)):
            r = compare_arrays(left, right, metrics=True, chunk_size=1)
            self.assertEqual(r['status'], 'completed')
            self.assertFalse(r['exact_equal'])
            self.assertEqual(r['dtype_differences'], 1)
            self.assertEqual(r['unequal_elements'], 1)
            self.assertEqual(r['metrics']['count'], 2)
            self.assertEqual(r['metrics']['max_abs']['value'],
                             abs(float(a['x'][1]) - float(b['x'][1])))
            self.assertEqual(r['metrics']['max_values']['dtype'],
                             dict(side_a=str(left['x'].dtype), side_b=str(right['x'].dtype)))
            json.dumps(r, allow_nan=False)
        equal = compare_arrays({'x': a['x'][:1]}, {'x': b['x'][:1]}, metrics=True)
        self.assertFalse(equal['exact_equal'])
        self.assertEqual(equal['unequal_elements'], 0)
        self.assertEqual(equal['metrics']['max_abs']['value'], 0.)

    def test_cross_dtype_integer_equality_precedes_lossy_metric_conversion(self):
        pairs = [(np.array([2**60+1], dtype=np.int64), np.array([2**60], dtype=np.uint64)),
                 (np.array([2**60], dtype=np.float64), np.array([2**60+1], dtype=np.int64))]
        for a, b in pairs:
            for x, y in ((a, b), (b, a)):
                with self.subTest(a=x.dtype, b=y.dtype):
                    r = compare_arrays({'x': x}, {'x': y}, metrics=True)
                    self.assertEqual(r['status'], 'completed')
                    self.assertEqual(r['unequal_elements'], 1)
                    self.assertFalse(r['exact_equal'])
                    self.assertIsNone(r['metrics'])
                    self.assertIn('integer', r['metrics_unavailable'])

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

    def test_exhaustive_findings_cover_all_coordinates_in_chunk_order(self):
        a = {'x': np.arange(24., dtype=np.float32).reshape(4, 6)[:, ::2]}
        b = {'x': a['x'] + 1}
        for chunk in (1, 5, 100):
            report = compare_arrays(a, b, max_findings=None, chunk_size=chunk)
            self.assertEqual(report['unequal_elements'], 12)
            self.assertEqual(len(report['findings']), 12)
            self.assertEqual(report['findings_omitted'], 0)
            for flat, finding in enumerate(report['findings']):
                self.assertEqual(finding['index'], list(np.unravel_index(flat, (4, 3))))
                self.assertEqual(finding['values']['side_a'], repr(a['x'].flat[flat].item()))
                self.assertEqual(finding['values']['side_b'], repr(b['x'].flat[flat].item()))
        a = {'x': np.array([np.nan, np.inf, 1., 2.])}
        b = {'x': np.array([np.nan, np.inf, 3., 2.])}
        report = compare_arrays(a, b, max_findings=None, chunk_size=2)
        self.assertEqual([row['index'] for row in report['findings']], [[0], [1], [2]])
        self.assertEqual([row['reason'] for row in report['findings']], ['nonfinite', 'nonfinite', 'values'])
        json.dumps(report, allow_nan=False)

    def test_groups_share_one_scan_and_match_direct_comparisons(self):
        from collections.abc import Mapping
        from collections import Counter
        class ReadOnce(Mapping):
            def __init__(self, values):
                self.values, self.reads = values, Counter()
            def __iter__(self):
                return iter(self.values)
            def __len__(self):
                return len(self.values)
            def __getitem__(self, key):
                self.reads[key] += 1
                if self.reads[key] > 1:
                    raise AssertionError('array loaded twice: ' + key)
                return self.values[key]
        values = {'a': np.arange(12).reshape(3, 4)[:, ::2],
                  'b': np.asarray(2.), 'c': np.arange(3.)}
        other = {k: v + 1 for k, v in values.items()}
        a, b = ReadOnce(values), ReadOnce(other)
        groups = {'gradients': ['a', 'b'], 'overlap': ['b', 'c'], 'single': ['a']}
        report = compare_arrays(a, b, groups=groups, metrics=True, chunk_size=2)
        self.assertEqual(dict(a.reads), dict.fromkeys(values, 1))
        self.assertEqual(dict(b.reads), dict.fromkeys(values, 1))
        for group, names in groups.items():
            expected = compare_arrays({k: values[k] for k in names},
                                      {k: other[k] for k in names}, metrics=True, chunk_size=2)
            self.assertEqual(report['groups'][group], expected)
        direct = compare_arrays(values, other, metrics=True, chunk_size=2)
        self.assertEqual({k: v for k, v in report.items() if k != 'groups'}, direct)

    def test_groups_keep_local_coverage_and_invalid_values(self):
        a = {'good': np.array([1.]), 'bad': np.array([np.nan]), 'shape': np.ones(2)}
        b = {'good': np.array([1.]), 'bad': np.array([np.nan]), 'shape': np.ones(3)}
        report = compare_arrays(a, b, expected=['good', 'bad', 'shape', 'absent'],
                                groups={k: [k] for k in ('good', 'bad', 'shape', 'absent')},
                                metrics=True)
        self.assertTrue(report['groups']['good']['exact_equal'])
        self.assertFalse(report['groups']['bad']['exact_equal'])
        self.assertIsNone(report['groups']['bad']['metrics'])
        self.assertEqual(report['groups']['shape']['status'], 'incomparable')
        self.assertEqual(report['groups']['absent']['status'], 'not_covered')
        self.assertEqual(report['groups']['absent']['coverage']['missing'],
                         dict(side_a=1, side_b=1))
        json.dumps(report, allow_nan=False)

    def test_groups_reject_invalid_memberships(self):
        a = {'x': np.ones(1)}
        for groups in ([], {'': ['x']}, {'g': []}, {'g': 'x'},
                       {'g': ['x', 'x']}, {'g': ['unknown']}, {1: ['x']}):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                compare_arrays(a, a, groups=groups)
        self.assertEqual(compare_arrays(a, a, groups={})['groups'], {})

    def assert_subset_oracle(self, report, a, b, names):
        pairs = [(a[k].ravel(), b[k].ravel()) for k in sorted(names)
                 if k in a and k in b and a[k].shape == b[k].shape]
        count = sum(x.size for x, _ in pairs)
        x = np.concatenate([x[np.isfinite(x) & np.isfinite(y)] for x, y in pairs])
        y = np.concatenate([y[np.isfinite(x) & np.isfinite(y)] for x, y in pairs])
        self.assertEqual(report['paired_finite_count'], x.size)
        self.assertEqual(report['paired_finite_fraction'], x.size / count if count else None)
        for side, arrays in (('side_a', a), ('side_b', b)):
            values = np.concatenate([arrays[k].ravel() for k in sorted(names) if k in arrays])
            self.assertEqual(report['counts'][side], dict(
                finite=int(np.isfinite(values).sum()), nan=int(np.isnan(values).sum()),
                posinf=int(np.isposinf(values).sum()), neginf=int(np.isneginf(values).sum())))
        metrics = report['finite_subset_metrics']
        self.assertIsNone(report['finite_subset_metrics_unavailable'])
        self.assertEqual(metrics['count'], x.size)
        for metric, value in dict(side_a_norm=np.linalg.norm(x), side_b_norm=np.linalg.norm(y),
                                  error_norm=np.linalg.norm(y - x), rmse=np.sqrt(np.mean((y - x)**2)),
                                  max_abs=np.max(np.abs(y - x)),
                                  cosine=np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))).items():
            self.assertAlmostEqual(metrics[metric]['value'], value)
        self.assertNotIn('max_flat_index', metrics)

    def test_nonfinite_missing_and_groups_match_independent_numpy_oracle(self):
        a = {'x': np.array([1., np.nan, np.inf, -np.inf, 5., 6.]),
             'y': np.array([2., 4.]), 'left': np.array([np.nan, np.inf, 7.]),
             'unexpected': np.array([np.nan])}
        b = {'x': np.array([2., 3., -np.inf, 4., np.nan, 9.]),
             'y': np.array([3., 8.]), 'right': np.array([-np.inf, 8.])}
        names = ['x', 'y', 'left', 'right', 'absent']
        groups = {'mixed': ['x', 'left', 'absent'], 'overlap': ['x', 'y'],
                  'missing': ['left', 'right']}
        report = compare_arrays(a, b, expected=names, groups=groups, mode='nonfinite',
                                metrics=True, chunk_size=2, max_findings=None)
        self.assertEqual(report['status'], 'not_covered')
        self.assertFalse(report['exact_equal'])
        self.assertEqual(report['coverage']['missing'], dict(side_a=2, side_b=2))
        self.assertEqual(report['coverage']['unexpected'], dict(side_a=1, side_b=0))
        self.assertEqual(report['coverage']['compared_elements'], 8)
        self.assertEqual(report['nonfinite'], dict(side_a=5, side_b=3))
        self.assertIsNone(report['metrics'])
        self.assert_subset_oracle(report, a, b, names)
        for group in ('mixed', 'overlap'):
            self.assert_subset_oracle(report['groups'][group], a, b, groups[group])
        for group, selected in groups.items():
            self.assertEqual(report['groups'][group], compare_arrays(
                {k: a[k] for k in selected if k in a}, {k: b[k] for k in selected if k in b},
                expected=selected, metrics=True, mode='nonfinite', chunk_size=2, max_findings=None))
        self.assertEqual(report['finite_subset_metrics']['max_coordinate'], dict(name='y', index=[1]))
        json.dumps(report, allow_nan=False)

    def test_nonfinite_one_sided_anomalies_have_native_positions_and_limits(self):
        a = {'left': np.array([[1., np.nan], [np.inf, -np.inf]], dtype=np.float32)}
        b = {'right': np.asarray(np.nan)}
        report = compare_arrays(a, b, mode='nonfinite', max_findings=None, chunk_size=1)
        anomalies = [f for f in report['findings'] if f['reason'] == 'nonfinite']
        self.assertEqual([(f['name'], f['side'], f['index']) for f in anomalies],
                         [('left', 'side_a', [0, 1]), ('left', 'side_a', [1, 0]),
                          ('left', 'side_a', [1, 1]), ('right', 'side_b', [])])
        self.assertEqual(anomalies[0]['values'], dict(dtype='float32', side_a='nan', side_b=None))
        self.assertEqual(anomalies[-1]['values'], dict(dtype='float64', side_a=None, side_b='nan'))
        self.assertEqual(report['counts'], dict(side_a=dict(finite=1, nan=1, posinf=1, neginf=1),
                                               side_b=dict(finite=0, nan=1, posinf=0, neginf=0)))
        self.assertFalse(report['exact_equal'])
        self.assertIsNone(report['paired_finite_fraction'])
        self.assertEqual(report['finite_subset_metrics']['count'], 0)
        for limit, omitted in ((0, 4), (3, 1), (8, 0)):
            bounded = compare_arrays(a, b, mode='nonfinite', max_findings=limit, chunk_size=2)
            self.assertEqual(bounded['findings_omitted'], omitted)
            self.assertEqual(bounded['counts'], report['counts'])
            self.assertLessEqual(len(bounded['findings']), limit)
        json.dumps(report, allow_nan=False)

    def test_nonfinite_maximum_keeps_original_coordinate_across_chunks(self):
        a = {'a': np.array([np.nan, 1.]),
             'z': np.arange(24., dtype=np.float32).reshape(4, 6)[:, ::2]}
        b = {k: v.copy() for k, v in a.items()}
        a['z'][0, 0] = np.nan
        b['z'][1, 0] = np.inf
        b['z'][2, 1] += 20
        b['z'][3, 0] += 20  # First original coordinate wins a tied maximum.
        for chunk in (1, 2, 5, 100):
            report = compare_arrays(a, b, mode='nonfinite', chunk_size=chunk,
                                    groups={'weights': ['z']})
            for result in (report, report['groups']['weights']):
                metrics = result['finite_subset_metrics']
                self.assertEqual(metrics['max_coordinate'], dict(name='z', index=[2, 1]))
                self.assertEqual(metrics['max_values'], dict(dtype='float32', side_a='14.0', side_b='34.0'))
            self.assert_subset_oracle(report, a, b, a)
        scalar = compare_arrays({'x': np.array(1)}, {'x': np.array(2)}, mode='nonfinite')
        self.assertEqual(scalar['finite_subset_metrics']['max_coordinate'], dict(name='x', index=[]))

    def test_nonfinite_zero_finite_coordinates_and_empty_arrays(self):
        for x, y in ((np.array([np.nan, np.inf]), np.array([1., -np.inf])),
                     (np.zeros(0), np.zeros(0))):
            report = compare_arrays({'x': x}, {'x': y}, mode='nonfinite', metrics=True)
            self.assertEqual(report['paired_finite_count'], 0)
            self.assertEqual(report['paired_finite_fraction'], 0. if x.size else None)
            self.assertEqual(report['finite_subset_metrics'],
                             dict(count=0, status='not_covered', reason='no present tensor coordinates'))
            self.assertIsNone(report['finite_subset_metrics_unavailable'])
            self.assertEqual(report['exact_equal'], not x.size)
            json.dumps(report, allow_nan=False)

    def test_nonfinite_shape_and_dtype_semantics_and_unpaired_validation(self):
        a = {'dtype': np.array([1., np.nan], dtype=np.float32),
             'shape': np.array([[np.inf, 2.]]), 'good': np.array([1., 3.])}
        b = {'dtype': np.array([2., 4.]), 'shape': np.array([-np.inf, np.nan, 3.]),
             'good': np.array([2., 5.])}
        report = compare_arrays(a, b, mode='nonfinite', metrics=True, max_findings=None,
                                groups={'shape': ['shape'], 'dtype': ['dtype']}, chunk_size=1)
        self.assertEqual(report['status'], 'incomparable')
        self.assertFalse(report['exact_equal'])
        self.assertEqual(report['dtype_differences'], 1)
        self.assertEqual(report['incompatible_arrays'], 1)
        self.assertEqual(report['coverage']['compared_elements'], 4)
        self.assertIsNone(report['metrics'])
        self.assert_subset_oracle(report, a, b, a)
        self.assertEqual([(f['side'], f['index']) for f in report['findings']
                          if f['name'] == 'shape' and f['reason'] == 'nonfinite'],
                         [('side_a', [0, 0]), ('side_b', [0]), ('side_b', [1])])
        self.assertIsNone(report['groups']['shape']['paired_finite_fraction'])
        self.assertFalse(report['groups']['dtype']['exact_equal'])
        self.assertEqual(report['groups']['dtype']['finite_subset_metrics']['count'], 1)
        for value in (np.array(['bad']), np.array([1j])):
            with self.assertRaisesRegex(ValueError, 'real numeric arrays'):
                compare_arrays({'x': value}, {}, mode='nonfinite')

    def test_nonfinite_preserves_integer_precision_and_overflow_limits(self):
        for x, y, reason in ((np.array([2**60+1], dtype=np.int64),
                              np.array([2**60], dtype=np.uint64), 'integer'),
                             (np.array([1e308]), np.array([-1e308]), 'nonfinite')):
            report = compare_arrays({'x': x}, {'x': y}, mode='nonfinite', metrics=True)
            self.assertFalse(report['exact_equal'])
            self.assertEqual(report['unequal_elements'], 1)
            self.assertEqual(report['paired_finite_count'], 1)
            self.assertIsNone(report['finite_subset_metrics'])
            self.assertIn(reason, report['finite_subset_metrics_unavailable'])
            self.assertIsNone(report['metrics'])
            self.assertEqual(report['findings'][-1]['values']['side_a'], repr(x[0].item()))
            json.dumps(report, allow_nan=False)
        x = {'x': np.array([-2**53, 2**53], dtype=np.int64)}
        report = compare_arrays(x, x, mode='nonfinite')
        self.assertTrue(report['exact_equal'])
        self.assertEqual(report['finite_subset_metrics']['count'], 2)

    def test_nonfinite_groups_reuse_native_loads_and_chunk_scans(self):
        from collections.abc import Mapping
        from collections import Counter
        class ReadOnce(Mapping):
            def __init__(self, values):
                self.values, self.reads = values, Counter()
            def __iter__(self):
                return iter(self.values)
            def __len__(self):
                return len(self.values)
            def __getitem__(self, key):
                self.reads[key] += 1
                if self.reads[key] > 1:
                    raise AssertionError('array loaded twice: ' + key)
                return self.values[key]
        a = ReadOnce({'x': np.array([1., np.nan, np.inf]), 'left': np.array([np.nan])})
        b = ReadOnce({'x': np.array([2., 3., np.inf])})
        from array_compare import _Comparison
        with patch.object(_Comparison, 'observe', autospec=True, side_effect=_Comparison.observe) as observe:
            compare_arrays(a, b, mode='nonfinite', groups={'one': ['x'], 'two': ['x', 'left']}, chunk_size=2)
        self.assertEqual(dict(a.reads), dict(x=1, left=1))
        self.assertEqual(dict(b.reads), dict(x=1))
        self.assertEqual(observe.call_count, 5)  # Two chunks per paired side; one unpaired chunk.

    def test_nonfinite_cli_and_default_schema_compatibility(self):
        a, b = {'x': np.array([1., np.nan])}, {'x': np.array([2., np.inf])}
        for metrics in (False, True):
            default = compare_arrays(a, b, metrics=metrics)
            self.assertEqual(default, compare_arrays(a, b, metrics=metrics, mode='metrics'))
            expected_keys = {'status', 'exact_equal', 'coverage', 'unequal_arrays', 'unequal_elements',
                             'incompatible_arrays', 'dtype_differences', 'nonfinite', 'findings', 'findings_omitted'}
            self.assertEqual(set(default), expected_keys | ({'metrics', 'metrics_unavailable'} if metrics else set()))
        finite = {'x': np.array([True, False]), 'y': np.array([1., 3.])}
        full = compare_arrays(finite, finite, metrics=True)
        diagnostic = compare_arrays(finite, finite, metrics=True, mode='nonfinite')
        self.assertEqual({k: diagnostic[k] for k in full}, full)
        self.assertEqual(diagnostic['finite_subset_metrics'],
                         {k: v for k, v in full['metrics'].items() if k != 'max_flat_index'})
        self.assertTrue(diagnostic['exact_equal'])
        for mode in ('unknown', '', None):
            with self.assertRaisesRegex(ValueError, 'mode must be'):
                compare_arrays(a, b, mode=mode)
        script = Path(__file__).resolve().parents[1] / 'skills/precision-compare/scripts/array_compare.py'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root/'a.npz', **a); np.savez(root/'b.npz', **b)
            (root/'groups.json').write_text('{"g": ["x"]}')
            (root/'expected.json').write_text('["x", "absent"]')
            argv = [sys.executable, '-I', '-B', str(script), str(root/'a.npz'), str(root/'b.npz'),
                    '--mode', 'nonfinite', '--all-findings', '--groups', str(root/'groups.json'),
                    '--expected', str(root/'expected.json'), '--output', str(root/'report.json')]
            process = subprocess.run(argv, capture_output=True, text=True)
            self.assertEqual(process.returncode, 1, process.stderr)
            report = json.loads((root/'report.json').read_text())
            self.assertEqual(report, compare_arrays(a, b, expected=['x', 'absent'],
                             groups={'g': ['x']}, mode='nonfinite', max_findings=None))
            self.assertNotIn('metrics', report)
            self.assertEqual(report['finite_subset_metrics']['count'], 1)
            argv[argv.index('nonfinite')] = 'unknown'
            self.assertEqual(subprocess.run(argv, capture_output=True).returncode, 2)

    def test_cli_groups_use_the_same_reports(self):
        script = Path(__file__).resolve().parents[1] / 'skills/precision-compare/scripts/array_compare.py'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a, b = {'x': np.array([1., 2.])}, {'x': np.array([1., 3.])}
            np.savez(root/'a.npz', **a); np.savez(root/'b.npz', **b)
            (root/'groups.json').write_text('{"step": ["x"]}')
            p = subprocess.run([sys.executable, '-I', '-B', str(script),
                                str(root/'a.npz'), str(root/'b.npz'), '--metrics',
                                '--groups', str(root/'groups.json'), '--output', str(root/'r.json')],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads((root/'r.json').read_text()),
                             compare_arrays(a, b, metrics=True, groups={'step': ['x']}))

    def test_invalid_inputs_and_inventories(self):
        for kwargs in ({'expected': ['x', 'x']}, {'expected': 'x'}, {'chunk_size': 0},
                       {'metrics': 1}, {'max_findings': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compare_arrays({'x': np.ones(1)}, {'x': np.ones(1)}, **kwargs)
        for a in ({'x': np.array(['x'])}, {'': np.ones(1)}, {'x': np.array([1j])}, {}):
            with self.subTest(a=a), self.assertRaises(ValueError):
                compare_arrays(a, a)

    def test_cli_exhaustive_findings_and_limit_are_explicit(self):
        script = Path(__file__).resolve().parents[1] / 'skills/precision-compare/scripts/array_compare.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / 'a.npz', x=np.zeros(12))
            np.savez(root / 'b.npz', x=np.ones(12))
            command = [sys.executable, '-B', str(script), str(root / 'a.npz'), str(root / 'b.npz'),
                       '--output', str(root / 'result.json'), '--all-findings']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / 'result.json').read_text())
            self.assertEqual(len(report['findings']), 12)
            self.assertEqual(subprocess.run(command + ['--max-findings', '1'], capture_output=True).returncode, 2)

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
