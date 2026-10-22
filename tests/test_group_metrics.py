"""Grouping preserves original metric arithmetic and explicit coordinate coverage."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'skills/precision-compare/scripts'))
from group_metrics import reduce_groups
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

    def test_invalid_coordinates_and_membership_are_not_silently_filtered(self):
        key = ('x', 0)
        cases = [([key, key], {}, {key: 0}),
                 ([key], {'group': [key, key]}, {key: 0}),
                 ([key], {'group': [('x', 1)]}, {key: 0}),
                 ([key], {}, {('x', 1): 0}),
                 ([('x', True)], {}, {('x', True): 0}),
                 ([], {}, {}), ([key], {'': [key]}, {key: 0})]
        for expected, groups, observations in cases:
            with self.subTest(expected=expected, groups=groups):
                with self.assertRaises(ValueError):
                    reduce_groups(expected, groups, lambda rows: 0, observations)
        with self.assertRaisesRegex(ValueError, 'finite'):
            reduce_groups([key], {}, reduce_scores, {key: float('nan')}, {key: 0.})


if __name__ == '__main__':
    unittest.main()
