"""Grouping preserves original metric arithmetic and explicit coordinate coverage."""
import copy
import json
from pathlib import Path
from collections.abc import Mapping
import sys
import tempfile

import numpy as np
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'skills/precision-compare/scripts'))
from group_metrics import reduce_groups
from array_compare import compare_arrays
from score_metrics import reduce_scores
from compare_distributions import summarize


class GroupMetricsTests(unittest.TestCase):
    def test_global_percentile_and_nll_use_targets_not_group_averages(self):
        keys = [('sample', i, 7) for i in range(4)]
        a = dict.fromkeys(keys, 0.)
        b = dict(zip(keys, [-1., -2., -3., -100.]))
        groups = {'short': keys[:3], 'long': keys[3:], 'overlap': keys[1:]}
        original = copy.deepcopy((a, b, groups))
        report = reduce_groups(iter(keys), groups, reduce_scores, a, b)
        self.assertEqual(report['global']['metrics'], reduce_scores(a, b))
        self.assertEqual(report['global']['metrics']['side_b_nll_nat_per_token'], 26.5)
        self.assertAlmostEqual(report['global']['metrics']['p99_nat'], 97.09)
        self.assertEqual(report['global']['metrics']['target_count'], 4)
        self.assertEqual(report['global']['metrics']['max_coordinate'], ['sample', 3, 7])
        self.assertEqual(report['groups']['overlap']['metrics']['target_count'], 3)
        self.assertEqual((a, b, groups), original)
        json.dumps(report, allow_nan=False)

    def test_incomplete_and_empty_groups_do_not_reduce_partial_intersections(self):
        keys = [('request', i) for i in range(3)]
        calls = []
        def reducer(a, b):
            calls.append(set(a))
            self.assertEqual(set(a), set(b))
            return len(a)
        report = reduce_groups(keys, {'ready': keys[:1], 'missing': keys[1:], 'empty': []},
                               reducer, {keys[0]: 1, keys[1]: 2}, {keys[0]: 3, keys[2]: 4})
        self.assertEqual(calls, [{keys[0]}])
        self.assertEqual(report['groups']['ready']['status'], 'completed')
        for result in [report['global'], report['groups']['missing'], report['groups']['empty']]:
            self.assertEqual(result['status'], 'not_covered')
            self.assertIsNone(result['metrics'])
        coverage = report['global']['coverage']
        self.assertEqual(coverage['observed_counts'], [2, 2])
        self.assertEqual(coverage['paired_count'], 1)
        self.assertEqual(coverage['missing'], [[['request', 2]], [['request', 1]]])

    def test_distribution_groups_preserve_summary_and_maximum_coordinates(self):
        def row(position, error):
            return dict(sample='x', position=position, probability_rmse=error/2,
                        max_probability_error=error, max_error_token=19,
                        tail_merged_js={'value': error/10}, top1_agreement=position != 0,
                        top100_overlap=.9, coverage={'side_a_union': .98, 'side_b_union': .97})
        rows = {('x', i): row(i, error) for i, error in [(2, .3), (0, .1), (1, .2)]}
        def reducer(values):
            return summarize(list(values.values()))
        report = reduce_groups(rows, {'one': [('x', 1)]}, reducer, rows)
        self.assertEqual(report['global']['metrics'], summarize([rows[k] for k in sorted(rows)]))
        self.assertEqual(report['groups']['one']['metrics'], summarize([rows['x', 1]]))
        self.assertEqual(report['global']['metrics']['max_probability_error']['max_coordinate'],
                         dict(sample='x', position=2, token=19))

    def test_arbitrary_coordinates_sources_and_stable_order(self):
        keys = [('weight', 2), ('weight', 1), ('loss', 'step-1')]
        sources = [dict.fromkeys(reversed(keys), i) for i in range(3)]
        report = reduce_groups(keys, {}, lambda *rows: [list(v) for v in rows], *sources)
        self.assertEqual(report['global']['coverage']['paired_count'], 3)
        self.assertTrue(all(order == report['global']['metrics'][0]
                            for order in report['global']['metrics']))
        self.assertEqual(report['groups'], {})

    def test_named_npz_groups_reuse_array_metrics_and_keep_element_coordinates(self):
        a = {'0/loss': np.asarray(1.), '0/update': np.asarray([1., 2., 3.]),
             '1/loss': np.asarray(2.), '1/update': np.asarray([4., 5., 6.])}
        b = {key: value.copy() for key, value in a.items()}
        b['1/update'][2] += 7.
        expected = list(a)
        groups = {'loss': ['0/loss', '1/loss'], 'update': ['0/update', '1/update'],
                  'first_step': ['0/loss', '0/update']}
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/name for name in ('a.npz', 'b.npz')]
            np.savez(paths[0], **a); np.savez(paths[1], **b)
            with np.load(paths[0]) as left, np.load(paths[1]) as right:
                report = reduce_groups(iter(expected), groups,
                    lambda x, y: compare_arrays(x, y, metrics=True), left, right)
        self.assertEqual(report['global']['metrics'], compare_arrays(a, b, metrics=True))
        self.assertEqual(report['global']['coverage']['paired_count'], 4)
        self.assertFalse(report['global']['metrics']['exact_equal'])
        for name, keys in groups.items():
            self.assertEqual(report['groups'][name]['metrics'], compare_arrays(
                {key: a[key] for key in keys}, {key: b[key] for key in keys}, metrics=True))
        self.assertTrue(report['groups']['first_step']['metrics']['exact_equal'])
        self.assertEqual(report['groups']['update']['metrics']['metrics']['max_coordinate'],
                         {'name': '1/update', 'index': [2]})
        json.dumps(report, allow_nan=False)

    def test_names_preserve_missing_inventory_including_shared_omissions(self):
        expected = ['loss', 'missing/both', 'missing/right']
        report = reduce_groups(expected, {'present': ['loss'], 'missing': expected[1:],
                                          'empty': []}, compare_arrays,
                               {'loss': np.asarray(1.), 'missing/right': np.asarray(2.)},
                               {'loss': np.asarray(1.)})
        self.assertEqual(report['global']['coverage']['missing'],
                         [['missing/both'], ['missing/both', 'missing/right']])
        self.assertEqual(report['global']['status'], 'not_covered')
        self.assertIsNone(report['global']['metrics'])
        self.assertTrue(report['groups']['present']['metrics']['exact_equal'])
        self.assertIsNone(report['groups']['missing']['metrics'])
        self.assertEqual(report['groups']['empty']['status'], 'not_covered')

    def test_mixed_names_and_tuples_are_distinct_and_serialize_missing_correctly(self):
        expected = ['weight', ('weight',)]
        report = reduce_groups(expected, {}, lambda values: list(values), {'weight': 1})
        self.assertEqual(report['global']['coverage']['missing'], [[['weight']]])
        self.assertEqual(report['global']['coverage']['paired_count'], 1)
        json.dumps(report, allow_nan=False)

    def test_nonfinite_named_group_does_not_hide_other_complete_group(self):
        values = {'finite': np.asarray([1., 2.]), 'invalid': np.asarray([np.nan])}
        report = reduce_groups(values, {'finite': ['finite'], 'invalid': ['invalid']},
                               compare_arrays, values, values)
        self.assertFalse(report['global']['metrics']['exact_equal'])
        self.assertTrue(report['groups']['finite']['metrics']['exact_equal'])
        self.assertFalse(report['groups']['invalid']['metrics']['exact_equal'])
        self.assertEqual(report['groups']['invalid']['metrics']['nonfinite'],
                         {'side_a': 1, 'side_b': 1})
        json.dumps(report, allow_nan=False)

    def test_group_views_are_lazy_read_only_and_restrict_coordinate_access(self):
        reads = []
        class Source(Mapping):
            def __iter__(self):
                return iter(['a', 'b'])
            def __len__(self):
                return 2
            def __getitem__(self, key):
                reads.append(key)
                return {'a': 1, 'b': 2}[key]
        calls = []
        def reducer(values):
            self.assertEqual(len(reads), len(calls))
            keys = list(values)
            self.assertEqual(len(values), len(keys))
            self.assertIn(keys[0], values)
            with self.assertRaises(KeyError):
                values['outside']
            if keys == ['b']:
                with self.assertRaises(KeyError):
                    values['a']
            with self.assertRaises(TypeError):
                values[keys[0]] = 0
            calls.append(keys)
            return values[keys[0]]
        report = reduce_groups(['b', 'a'], {'second': ['b']}, reducer, Source())
        self.assertEqual(calls, [['a', 'b'], ['b']])
        self.assertEqual(reads, ['a', 'b'])
        self.assertEqual(report['global']['metrics'], 1)
        self.assertEqual(report['groups']['second']['metrics'], 2)

    def test_invalid_coordinates_and_membership_are_not_silently_filtered(self):
        key = ('x', 0)
        cases = [([key, key], {}, {key: 0}),
                 ([key], {'group': [key, key]}, {key: 0}),
                 ([key], {'group': [('x', 1)]}, {key: 0}),
                 ([key], {}, {('x', 1): 0}),
                 ([('x', True)], {}, {('x', True): 0}),
                 ([], {}, {}), ([key], {'': [key]}, {key: 0}),
                 ('loss', {}, {'loss': 0}), (['loss'], {'g': 'loss'}, {'loss': 0}),
                 (['loss', 'loss'], {}, {'loss': 0}), ([''], {}, {'': 0}),
                 (['loss'], {'g': ['loss', 'loss']}, {'loss': 0}),
                 (['loss'], {'g': ['other']}, {'loss': 0}),
                 (['loss'], {}, {'other': 0})]
        for expected, groups, observations in cases:
            with self.subTest(expected=expected, groups=groups):
                with self.assertRaises(ValueError):
                    reduce_groups(expected, groups, lambda rows: 0, observations)
        with self.assertRaisesRegex(ValueError, 'finite'):
            reduce_groups([key], {}, reduce_scores, {key: float('nan')}, {key: 0.})


if __name__ == '__main__':
    unittest.main()
