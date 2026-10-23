"""Nested comparisons preserve typed structure and existing numeric arithmetic."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'skills/precision-compare/scripts'))
from tree_compare import compare_trees
from array_compare import compare_arrays


class TreeCompareTests(unittest.TestCase):
    def test_mapping_order_is_irrelevant_and_shared_subtrees_are_allowed(self):
        leaf = {'weight': np.arange(6, dtype=np.float32).reshape(2, 3)}
        a = {'model': leaf, 'alias': leaf, 'state': [None, 'adam', (1, True, .5)]}
        b = {'state': [None, 'adam', (1, True, .5)], 'alias': copy.deepcopy(leaf),
             'model': copy.deepcopy(leaf)}
        report = compare_trees(a, b, metrics=True)
        self.assertTrue(report['exact_equal'])
        self.assertEqual(report['metadata']['differences'], 0)
        self.assertEqual(report['coverage']['missing_nodes'], {'side_a': 0, 'side_b': 0})
        self.assertEqual(report['numeric']['metrics']['count'], 15)
        np.testing.assert_array_equal(leaf['weight'], np.arange(6).reshape(2, 3))
        json.dumps(report, allow_nan=False)

    def test_typed_keys_and_paths_do_not_collide(self):
        a = {1: np.asarray([1.]), '1': np.asarray([2.]), 'a/b': np.asarray([3.]),
             'a': {'b': np.asarray([4.])}}
        b = copy.deepcopy(a); b['a/b'][0] += 7
        report = compare_trees(a, b, metrics=True)
        self.assertFalse(report['exact_equal'])
        self.assertEqual(report['numeric']['coverage']['paired_arrays'], 4)
        self.assertEqual(report['numeric']['metrics']['max_abs']['value'], 7)
        self.assertEqual(report['numeric']['metrics']['max_coordinate']['path'],
                         [['key', 'str', 'a/b']])
        changed_key = compare_trees({1: 'value'}, {'1': 'value'})
        self.assertFalse(changed_key['exact_equal'])
        self.assertEqual(changed_key['coverage']['missing_nodes'], {'side_a': 1, 'side_b': 1})

    def test_sequence_types_and_missing_empty_containers_are_visible(self):
        report = compare_trees({'values': [1, 2], 'empty': {}}, {'values': (1, 2)})
        self.assertFalse(report['exact_equal'])
        self.assertEqual(report['metadata']['differences'], 2)
        self.assertEqual(report['coverage']['missing_nodes']['side_b'], 1)
        self.assertTrue(report['numeric']['exact_equal'])
        self.assertFalse(compare_trees({'x': None}, {})['exact_equal'])
        self.assertFalse(compare_trees([], ())['exact_equal'])

    def test_array_shape_dtype_and_nonfinite_checks_are_reused(self):
        for peer in (np.asarray([1., 2.], dtype=np.float64),
                     np.asarray([[1., 2.]], dtype=np.float32)):
            report = compare_trees(np.asarray([1., 2.], dtype=np.float32), peer)
            self.assertFalse(report['exact_equal'])
            self.assertEqual(report['numeric']['incompatible_arrays'], int(peer.shape != (2,)))
            self.assertEqual(report['numeric']['dtype_differences'], int(peer.shape == (2,)))
        report = compare_trees({'x': np.asarray([np.nan, np.inf])},
                               {'x': np.asarray([np.nan, np.inf])}, metrics=True)
        self.assertFalse(report['exact_equal'])
        self.assertEqual(report['numeric']['nonfinite'], {'side_a': 2, 'side_b': 2})
        self.assertIsNone(report['numeric']['metrics'])
        json.dumps(report, allow_nan=False)

    def test_numeric_reduction_uses_existing_metrics_and_element_coordinates(self):
        a = [np.asarray([1., 2.]), np.asarray([3., 4., 5.])]
        b = [np.asarray([1., 3.]), np.asarray([3., 4., 8.])]
        report = compare_trees(a, b, metrics=True)
        direct = compare_arrays({'a': a[0], 'b': a[1]}, {'a': b[0], 'b': b[1]}, metrics=True)
        for key in ('side_a_norm', 'side_b_norm', 'error_norm', 'rmse', 'cosine', 'max_abs'):
            self.assertEqual(report['numeric']['metrics'][key], direct['metrics'][key])
        coordinate = report['numeric']['metrics']['max_coordinate']
        self.assertEqual(coordinate['path'], [['index', 1]])
        self.assertEqual(coordinate['index'], [2])

    def test_adapter_preserves_native_identity_without_a_framework_dependency(self):
        class Tensor:
            def __init__(self, dtype):
                self.dtype = dtype
        def adapt(value):
            return {'native_dtype': value.dtype, 'values': np.asarray([1., 2.], dtype=np.float32)}
        self.assertTrue(compare_trees(Tensor('bf16'), Tensor('bf16'), leaf_adapter=adapt)['exact_equal'])
        report = compare_trees(Tensor('bf16'), Tensor('fp16'), leaf_adapter=adapt)
        self.assertFalse(report['exact_equal'])
        self.assertTrue(report['numeric']['exact_equal'])
        with self.assertRaises(TypeError):
            compare_trees(Tensor('bf16'), Tensor('bf16'))
        with self.assertRaises(TypeError):
            compare_trees(Tensor('bf16'), Tensor('bf16'), leaf_adapter=lambda value: value)

    def test_adapter_results_cannot_trigger_unbounded_re_adaptation(self):
        for adapter in (lambda value: object(), lambda value: {'new': object()}):
            with self.subTest(adapter=adapter):
                with self.assertRaisesRegex(TypeError, 'unsupported state'):
                    compare_trees(object(), {}, leaf_adapter=adapter)

    def test_cycles_are_rejected_and_adapter_cycles_do_not_recurse_forever(self):
        a = []; a.append(a)
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            compare_trees(a, [])
        value = object()
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            compare_trees(value, {}, leaf_adapter=lambda item: {'self': item})

    def test_metadata_only_empty_states_and_large_integers_remain_exact(self):
        for value in ({}, [], (), None, 'value', 10**100):
            report = compare_trees(value, copy.deepcopy(value), metrics=True)
            self.assertTrue(report['exact_equal'])
            self.assertIsNone(report['numeric'])
            json.dumps(report, allow_nan=False)
        self.assertFalse(compare_trees(10**100, 10**100+1)['exact_equal'])
        self.assertFalse(compare_trees('1', 1)['exact_equal'])
        empty = compare_trees(np.asarray([]), np.asarray([]), metrics=True)
        self.assertTrue(empty['exact_equal'])
        self.assertEqual(empty['numeric']['coverage']['compared_elements'], 0)

    def test_bounded_findings_retain_complete_counts(self):
        a = {str(i): i for i in range(20)}
        b = {str(i): i+1 for i in range(20)}
        a['text'] = 'old'; b['text'] = 'new'
        report = compare_trees(a, b, max_findings=0)
        self.assertFalse(report['exact_equal'])
        self.assertEqual(report['numeric']['unequal_arrays'], 20)
        self.assertEqual(report['numeric']['findings'], [])
        self.assertEqual(report['numeric']['findings_omitted'], 20)
        self.assertEqual(report['metadata']['differences'], 1)
        self.assertEqual(report['metadata']['findings_omitted'], 1)

    def test_checkpoint_reports_scan_and_convert_each_tensor_once(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional Torch checkpoint loader')
        import tempfile
        from unittest.mock import patch
        from tree_compare import compare_checkpoints
        from compare_tensors import Stats
        a = {'model': torch.arange(8, dtype=torch.bfloat16),
             'optimizer': {0: {'moment': torch.ones(4, dtype=torch.bfloat16)}},
             'empty': {}, 1: 'typed key', '1': None}
        b = copy.deepcopy(a)
        b['optimizer'][0]['moment'][2] += .25
        b.pop('empty')
        b['1'] = 'changed'
        conversions, scanned = [], []
        original_float, original_add = torch.Tensor.float, Stats.add
        def converted(tensor, *args, **kwargs):
            conversions.append(tensor.numel())
            return original_float(tensor, *args, **kwargs)
        def added(stats, x, y):
            scanned.append(x.size)
            return original_add(stats, x, y)
        def export(tensor):
            return dict(native_dtype=str(tensor.dtype), values=original_float(tensor).numpy())
        expected = compare_trees(a, b, leaf_adapter=export, metrics=True)
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / (name + '.pt') for name in ('a', 'b')]
            for path, state in zip(paths, (a, b)):
                torch.save(state, path)
            with patch.object(torch.Tensor, 'float', converted), patch.object(Stats, 'add', added):
                report = compare_checkpoints(*paths, metrics=True)
        self.assertEqual(sum(conversions), 24)  # 12 elements per side, one conversion.
        self.assertEqual(sum(scanned), 12)
        self.assertEqual(report['comparison'], expected)
        for section in report['sections']:
            key = section['key']
            peers = [{key: state[key]} if key in state else {} for state in (a, b)]
            self.assertEqual(section['comparison'],
                             compare_trees(*peers, leaf_adapter=export, metrics=True))

    def test_checkpoint_sections_retain_independent_coverage_and_metric_failures(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional Torch checkpoint loader')
        import tempfile
        from tree_compare import compare_checkpoints
        a = {'good': np.array([1., 2.]), 'nonfinite': np.array([np.nan]),
             'shape': np.ones(2), 'missing': np.ones(1), 'empty': {}}
        b = {'good': np.array([1., 3.]), 'nonfinite': np.array([np.inf]),
             'shape': np.ones((1, 2)), 'empty': {}}
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / (name + '.pt') for name in ('a', 'b')]
            for path, state in zip(paths, (a, b)):
                torch.save(state, path)
            for metrics, limit in ((True, 8), (True, 0), (False, 1)):
                with self.subTest(metrics=metrics, max_findings=limit):
                    options = dict(metrics=metrics, max_findings=limit)
                    report = compare_checkpoints(*paths, **options)
                    self.assertEqual(report['comparison'], compare_trees(a, b, **options))
                    for section in report['sections']:
                        key = section['key']
                        peers = [{key: state[key]} if key in state else {} for state in (a, b)]
                        self.assertEqual(section['comparison'], compare_trees(*peers, **options))
                    if metrics:
                        sections = {row['key']: row['comparison'] for row in report['sections']}
                        self.assertEqual(sections['good']['numeric']['metrics']['max_abs']['value'], 1.)
                        self.assertIsNone(report['comparison']['numeric']['metrics'])

    def test_checkpoint_cli_reuses_tree_reports_and_keeps_native_dtype(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional Torch checkpoint loader')
        import hashlib
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = {'model': torch.tensor([1., 2.], dtype=torch.bfloat16),
                 'optimizer': {0: {'moment': torch.tensor([.25, .5])}},
                 'rng': np.arange(4, dtype=np.uint32), 'step': 3, 'empty': {}}
            b = copy.deepcopy(a);b['optimizer'][0]['moment'][1] += .25
            b['step'] = 4;b.pop('empty')
            for name, state in (('a.pt', a), ('b.pt', b)):
                torch.save(state, root/name)
            hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('a.pt', 'b.pt')}
            script = Path(__file__).resolve().parents[1]/'skills/precision-compare/scripts/tree_compare.py'
            command = [sys.executable, '-I', '-B', str(script), str(root/'a.pt'),
                       str(root/'b.pt'), '--output', str(root/'result.json'), '--metrics']
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((root/'result.json').read_text())
            self.assertFalse(report['comparison']['exact_equal'])
            self.assertEqual(report['comparison']['numeric']['unequal_elements'], 2)
            sections = {row['key']: row['comparison'] for row in report['sections']}
            self.assertTrue(sections['model']['exact_equal'])
            self.assertTrue(sections['rng']['exact_equal'])
            self.assertFalse(sections['empty']['exact_equal'])
            self.assertEqual(sections['empty']['coverage']['missing_nodes']['side_b'], 1)
            finding = sections['optimizer']['numeric']['findings'][0]
            self.assertEqual(finding['index'], [1])
            self.assertEqual(finding['values']['side_b'], '0.75')
            for side, name in (('side_a', 'a.pt'), ('side_b', 'b.pt')):
                self.assertEqual(report['inputs'][side]['sha256'], hashes[name])
                self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(), hashes[name])
            before = (root/'result.json').read_bytes()
            again = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(again.returncode, 2)
            self.assertEqual((root/'result.json').read_bytes(), before)
            (root/'bad.pt').write_bytes(b'not a checkpoint')
            invalid = subprocess.run([sys.executable, '-I', '-B', str(script),
                                     str(root/'bad.pt'), str(root/'b.pt'), '--output',
                                     str(root/'invalid.json')], capture_output=True, text=True)
            self.assertEqual(invalid.returncode, 2)
            self.assertFalse((root/'invalid.json').exists())
            from tree_compare import compare_checkpoints
            torch.save(torch.tensor([1., 2.], dtype=torch.float32), root/'b.pt')
            torch.save(torch.tensor([1., 2.], dtype=torch.bfloat16), root/'a.pt')
            report = compare_checkpoints(root/'a.pt', root/'b.pt')
            self.assertFalse(report['comparison']['exact_equal'])
            self.assertTrue(report['comparison']['numeric']['exact_equal'])
            self.assertEqual(report['sections'], [])

    def test_invalid_options_keys_and_array_types_are_explicit(self):
        for kwargs in ({'metrics': 1}, {'max_findings': True}, {'max_findings': -1},
                       {'leaf_adapter': 1}):
            with self.assertRaises(ValueError):
                compare_trees({}, {}, **kwargs)
        for value in ({True: 1}, {(1,): 2}, object(), np.asarray(['x']), np.asarray([1j])):
            with self.assertRaises(TypeError):
                compare_trees(value, value)


if __name__ == '__main__':
    unittest.main()
