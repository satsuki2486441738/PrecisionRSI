"""Nonfinite diagnostics regressions using the public tensor artifact contract."""
import copy
from functools import partial
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills/precision-compare/scripts'
sys.path.insert(0, str(SCRIPTS))
from artifact_io import Array
from compare_tensors import CATEGORIES, compare as finite_compare

compare = partial(finite_compare, mode='nonfinite')


class NonfiniteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.serial = 0
        self.common = dict(model='model-v1', inputs='input-v1', numerics='fp64',
                           scope='test', state='initial-v1')
        for side in ('a', 'b'):
            self.write(f'{side}-env.json', dict(rank=0, versions={'numpy': np.__version__}))
            self.write(f'{side}-status.json', dict(rank=0, status='completed'))

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, allow_nan=False))
        return path

    def array(self, values, dtype=np.float64):
        self.serial += 1
        path = self.root / f'array-{self.serial}.npy'
        np.save(path, np.asarray(values, dtype=dtype))
        return {'path': path.name}

    def pair(self, a, b, stage='gradients', shape=None):
        entry = dict(name='weight', shape=shape or [len(a)], present=True)
        spec = dict(schema_version=1, kind='tensors', common=self.common, thresholds=None,
                    expected_ranks={'a': [0], 'b': [0]}, stage=stage, tensors=[entry])
        manifests = []
        if stage == 'observations':
            entry['coordinates'] = dict(sample='sample-2', boundary='logits', invocation=3)
        for side, values in (('a', a), ('b', b)):
            manifests.append(dict(schema_version=1, kind='tensors', common=copy.deepcopy(self.common),
                side=side, stage=stage, expected_ranks=[0],
                ranks=[dict(rank=0, environment=f'{side}-env.json', status=f'{side}-status.json')],
                events={'actor': 'applied'},
                tensors=[dict(copy.deepcopy(entry), shards=[dict(offset=0, value=self.array(values))])]))
        return spec, *manifests

    def paths(self, spec, a, b):
        return (self.write('contract.json', spec), self.write('a.json', a), self.write('b.json', b))

    def run_pair(self, pair, **kwargs):
        result = compare(*self.paths(*pair), **kwargs)
        json.dumps(result, allow_nan=False)
        return result

    def test_all_joint_categories_and_signs(self):
        values = [1., np.nan, np.inf, -np.inf]
        a = np.repeat(values, 4)
        b = np.tile(values, 4)
        result = self.run_pair(self.pair(a, b), chunk_size=3)
        self.assertEqual(result['status'], 'no_thresholds')
        g = result['global']
        self.assertEqual(g['joint_counts'], {f'{x}/{y}': 1 for x in CATEGORIES for y in CATEGORIES})
        self.assertEqual(g['side_a'], dict(finite=4, nan=4, posinf=4, neginf=4))
        self.assertEqual(g['side_b'], g['side_a'])
        self.assertEqual(g['paired_finite_count'], 1)
        self.assertEqual(g['paired_finite_fraction'], 1/16)
        self.assertEqual(g['finite_subset_metrics']['rmse']['value'], 0.)

    def test_independent_shards_samples_and_finite_coordinate_mapping(self):
        pair = self.pair([np.nan, 1, np.inf, 2, -np.inf, 5],
                         [0, 4, np.inf, 3, np.nan, 1], 'observations', [2, 3])
        spec, a, b = pair
        a['tensors'][0]['shards'] = [dict(offset=0, value=self.array([np.nan, 1])),
                                    dict(offset=2, value=self.array([np.inf, 2, -np.inf, 5]))]
        b['tensors'][0]['shards'] = [dict(offset=0, value=self.array([0])),
                                    dict(offset=1, value=self.array([4, np.inf, 3, np.nan, 1]))]
        reports = [self.run_pair(pair, chunk_size=n, max_locations=2) for n in (1, 2, 4, 100)]
        for result in reports:
            g = result['global']
            self.assertEqual(g['paired_finite_count'], 3)
            self.assertEqual([s['flat_index'] for s in g['samples']], [0, 2])
            maximum = g['finite_subset_metrics']['max_coordinate']
            self.assertEqual(maximum['flat_index'], 5)
            self.assertEqual(maximum['coordinate'], [1, 2])
            self.assertEqual(maximum['observation'], spec['tensors'][0]['coordinates'])
            self.assertAlmostEqual(g['finite_subset_metrics']['rmse']['value'], np.sqrt(26/3))
            self.assertNotIn('max_flat_index', g['finite_subset_metrics'])
            self.assertEqual(g['samples'], reports[0]['global']['samples'])
            self.assertEqual(g['joint_counts'], reports[0]['global']['joint_counts'])
        # A later, smaller finite error must not replace the maximum; ties retain first.
        result = self.run_pair(self.pair([np.nan, 1, 1, 1], [0, 5, 5, 2]), chunk_size=1)
        self.assertEqual(result['global']['finite_subset_metrics']['max_coordinate']['flat_index'], 1)

    def test_no_paired_finite(self):
        g = self.run_pair(self.pair([np.nan, 1], [0, np.inf]))['global']
        self.assertEqual(g['paired_finite_count'], 0)
        self.assertEqual(g['finite_subset_metrics']['reason'], 'no paired finite coordinates')
        self.assertEqual(g['count'], 2)

    def test_absent_gradients_and_missing_coverage(self):
        pair = self.pair([1], [2])
        spec, a, b = pair
        for doc in pair:
            doc['tensors'].append(dict(name='unused', shape=[10], present=False,
                **({} if doc is spec else {'shards': []})))
        result = self.run_pair(pair)
        self.assertEqual(result['coverage']['absent_tensors'], ['unused'])
        self.assertEqual(result['global']['count'], 1)
        b['tensors'].pop()
        result = self.run_pair(pair)
        self.assertEqual(result['status'], 'not_covered')
        self.assertEqual(result['coverage']['invalid_tensors'], ['unused'])
        self.assertEqual(result['per_tensor']['unused']['side_a']['presence'], 'absent')
        self.assertNotIn('joint_counts', result['global'])

    def test_all_absent(self):
        pair = self.pair([1], [2])
        for doc in pair:
            doc['tensors'][0]['present'] = False
            if 'side' in doc:
                doc['tensors'][0]['shards'] = []
        result = self.run_pair(pair)
        self.assertEqual(result['status'], 'no_thresholds')
        self.assertIsNone(result['global']['paired_finite_fraction'])
        self.assertEqual(result['global']['count'], 0)

    def test_invalid_artifacts_keep_valid_side_and_other_tensor(self):
        for mutation in ('missing', 'gap', 'dtype', 'shape', 'presence', 'coordinate'):
            with self.subTest(mutation=mutation):
                pair = self.pair([np.nan, 2], [1, 2], 'observations')
                spec, a, b = pair
                for doc in pair:
                    entry = copy.deepcopy(doc['tensors'][0])
                    entry['name'] = 'good'
                    doc['tensors'].append(entry)
                entry = b['tensors'][0]
                if mutation == 'missing':
                    entry['shards'][0]['value']['path'] = 'missing.npy'
                elif mutation == 'gap':
                    entry['shards'][0]['offset'] = 1
                elif mutation == 'dtype':
                    entry['shards'][0]['value'] = self.array([1, 2], np.int32)
                elif mutation == 'shape':
                    entry['shape'] = [1, 2]
                elif mutation == 'presence':
                    entry['present'] = False
                else:
                    entry['coordinates']['sample'] = 'wrong'
                result = self.run_pair(pair)
                self.assertNotEqual(result['status'], 'no_thresholds')
                self.assertEqual(result['coverage']['invalid_tensors'], ['weight'])
                self.assertEqual(result['coverage']['diagnosed_tensors'], ['good'])
                self.assertEqual(result['per_tensor']['weight']['side_a']['counts']['nan'], 1)
                self.assertNotIn('count', result['global'])

    def test_unexpected_duplicate_and_bad_conditions(self):
        pair = self.pair([1], [1])
        extra = copy.deepcopy(pair[2]['tensors'][0])
        extra['name'] = 'extra'
        pair[2]['tensors'].append(extra)
        result = self.run_pair(pair)
        self.assertEqual(result['coverage']['unexpected_tensors']['side_b'], ['extra'])
        self.assertEqual(result['global']['status'], 'not_covered')
        pair[2]['tensors'][1]['name'] = 'weight'
        self.assertIn('duplicate', self.run_pair(pair)['reason'])
        pair = self.pair([1], [1])
        pair[2]['common']['model'] = 'wrong'
        self.assertEqual(self.run_pair(pair)['status'], 'incomparable')

    def test_finite_subtraction_overflow_preserves_all_counts(self):
        largest = np.finfo(np.float64).max
        pair = self.pair([largest, np.nan, 2, -np.inf], [-largest, 1, 3, np.inf])
        for n in (1, 2, 100):
            result = self.run_pair(pair, chunk_size=n)
            g = result['global']
            self.assertEqual(result['status'], 'no_thresholds')
            self.assertEqual(g['count'], 4)
            self.assertEqual(g['paired_finite_count'], 2)
            self.assertEqual(g['side_a']['nan'], 1)
            self.assertEqual(g['joint_counts']['neginf/posinf'], 1)
            self.assertIn('nonfinite tensor difference', g['finite_subset_metrics']['reason'])
            self.assertIn('nonfinite tensor difference', result['per_tensor']['weight']['finite_subset_metrics']['reason'])

    def test_norm_overflow_is_explicit_finite_json(self):
        largest = np.finfo(np.float64).max
        g = self.run_pair(self.pair([largest, largest], [largest, largest]))['global']
        self.assertIsNone(g['finite_subset_metrics']['side_a_norm']['value'])
        self.assertEqual(g['finite_subset_metrics']['rmse']['value'], 0.)

    def test_updates_stored_invalid_vs_generated_delta(self):
        largest = np.finfo(np.float64).max
        pair = self.pair([0]*5, [0]*5, 'updates')
        for m in pair[1:]:
            m['tensors'][0]['shards'] = [dict(offset=0,
                before=self.array([np.nan, np.inf, -largest, 1]),
                after=self.array([1, np.inf, largest, 2])),
                dict(offset=4, value=self.array([-np.inf]))]
        for n in (1, 3, 99):
            result = self.run_pair(pair, chunk_size=n)
            side = result['per_tensor']['weight']['side_a']
            self.assertEqual(side['stored_before'], dict(finite=2, nan=1, posinf=1, neginf=0))
            self.assertEqual(side['stored_after'], dict(finite=3, nan=0, posinf=1, neginf=0))
            self.assertEqual(side['before_after_count'], 4)
            self.assertEqual(side['generated_nonfinite_delta'], 1)
            self.assertEqual(side['counts'], dict(finite=1, nan=2, posinf=1, neginf=1))
            self.assertEqual(result['global']['paired_finite_count'], 1)
            self.assertEqual(result['global']['finite_subset_metrics']['rmse']['value'], 0.)
        del pair[2]['events']
        self.assertEqual(self.run_pair(pair)['status'], 'not_covered')

    def test_promoted_updates_and_declared_state_only(self):
        pair = self.pair([0, 0], [0, 0], 'updates')
        for m, before in zip(pair[1:], ([1, 2], [3, 4])):
            m['tensors'][0]['shards'] = [dict(offset=0, before=self.array(before, np.float16),
                after=self.array(np.asarray(before)+.125, np.float16))]
        g = self.run_pair(pair)['global']
        self.assertEqual(g['finite_subset_metrics']['side_a_norm']['value'], np.sqrt(2)*.125)
        self.assertEqual(g['finite_subset_metrics']['rmse']['value'], 0.)

    def test_invalid_parameters_and_zero_sample_limit(self):
        pair = self.pair([np.nan], [np.inf])
        for parameter in ('chunk_size', 'max_locations'):
            for bad in (-1, True, 1.5, None, '2'):
                with self.subTest(parameter=parameter, bad=bad):
                    result = self.run_pair(pair, **{parameter: bad})
                    self.assertEqual(result['status'], 'incomparable')
                    self.assertIn(parameter, result['reason'])
        self.assertEqual(self.run_pair(pair, chunk_size=0)['status'], 'incomparable')
        self.assertEqual(self.run_pair(pair, max_locations=0)['global']['samples'], [])

    def test_streaming_reads_bounded_and_chunk_invariant(self):
        a = np.arange(5003, dtype=float)
        b = a + 1
        a[::31] = np.nan
        b[::73] = -np.inf
        pair = self.pair(a, b)
        original = Array.read
        reads = []
        def bounded(reader, start, count):
            reads.append(count)
            self.assertLessEqual(count, 37)
            return original(reader, start, count)
        with patch.object(Array, 'read', bounded):
            small = self.run_pair(pair, chunk_size=37, max_locations=4)
        self.assertGreater(len(reads), 100)
        large = self.run_pair(pair, chunk_size=2000, max_locations=4)
        for key in ('side_a', 'side_b', 'joint_counts', 'samples', 'paired_finite_count', 'count'):
            self.assertEqual(small['global'][key], large['global'][key])

    def test_bf16_safetensors(self):
        pair = self.pair([0]*4, [0]*4)
        header = json.dumps({'x': dict(dtype='BF16', shape=[4], data_offsets=[0, 8])}).encode()
        (self.root/'values.safetensors').write_bytes(struct.pack('<Q', len(header)) + header +
                                                   struct.pack('<4H', 0x3f80, 0x7fc0, 0x7f80, 0xff80))
        pair[1]['tensors'][0]['shards'][0]['value'] = dict(path='values.safetensors', key='x')
        self.assertEqual(self.run_pair(pair)['global']['side_a'], dict(finite=1, nan=1, posinf=1, neginf=1))

    def test_schema2_actual_conditions_and_self_comparison(self):
        pair = self.pair([np.nan, 1], [np.inf, 2])
        for doc in pair:
            doc['schema_version'] = 2
        for side in ('a', 'b'):
            self.write(f'{side}-env.json', dict(rank=0, versions={'numpy': np.__version__}, actual=self.common))
        paths = self.paths(*pair)
        self.assertEqual(compare(*paths)['status'], 'no_thresholds')
        result = compare(paths[0], paths[1], paths[1])
        self.assertTrue(result['self_comparison'])
        self.assertEqual(result['global']['joint_counts']['nan/nan'], 1)
        self.write('b-env.json', dict(rank=0, versions={'numpy': np.__version__}))
        self.assertEqual(compare(*paths)['status'], 'not_covered')

    def test_cli_strict_json_exit_codes_and_output_preservation(self):
        paths = self.paths(*self.pair([np.nan, 1], [np.inf, 2]))
        output = self.root/'report.json'
        command = [sys.executable, '-B', str(SCRIPTS/'compare_tensors.py'), '--mode', 'nonfinite',
                   '--contract', str(paths[0]), '--side-a', str(paths[1]), '--side-b', str(paths[2]),
                   '--output', str(output), '--chunk-size', '1', '--max-locations', '1']
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(output.read_text(), parse_constant=lambda s: self.fail(s))
        self.assertEqual(len(result['global']['samples']), 1)
        original = output.read_bytes()
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
        self.assertEqual(output.read_bytes(), original)
        output.unlink()
        command[-1] = '-1'
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
        self.assertEqual(json.loads(output.read_text())['status'], 'incomparable')

    def test_tensor_modes_preserve_nonfinite_coverage_and_default_failure(self):
        paths = self.paths(*self.pair([np.nan, 1., -np.inf], [np.inf, 3., -np.inf],
                                    stage='observations'))
        default = finite_compare(*paths)
        self.assertEqual(default, finite_compare(*paths, mode='metrics'))
        self.assertNotEqual(default['status'], 'no_thresholds')
        for chunk_size in (1, 2, 20):
            for limit in (0, 1, 20):
                with self.subTest(chunk_size=chunk_size, limit=limit):
                    options = dict(chunk_size=chunk_size, max_locations=limit)
                    result = finite_compare(*paths, mode='nonfinite', **options)
                    self.assertEqual(result['global']['count'], 3)
                    self.assertEqual(result['global']['paired_finite_count'], 1)
                    self.assertEqual(result['global']['finite_subset_metrics']['rmse']['value'], 2.)
                    self.assertLessEqual(len(result['global']['samples']), limit)

    def test_both_modes_validate_update_events_before_reading_arrays(self):
        variants = [({'actor': 'applied'}, {'actor': 'inactive'}, 'incomparable'),
                    ({'actor': 'inactive'}, {'actor': 'inactive'}, 'not_covered'),
                    ({'actor': 'applied'}, {'other': 'applied'}, 'incomparable'),
                    ({'actor': 'applied'}, {'actor': 'invalid'}, 'incomparable'),
                    ({'actor': 'applied'}, {}, 'not_covered')]
        for a_events, b_events, status in variants:
            pair = self.pair([1., 2.], [1., 2.], 'updates')
            pair[1]['events'], pair[2]['events'] = a_events, b_events
            paths = self.paths(*pair)
            for mode in ('metrics', 'nonfinite'):
                with self.subTest(mode=mode, events=(a_events, b_events)):
                    with patch.object(Array, 'read', side_effect=AssertionError('unexpected array read')):
                        result = finite_compare(*paths, mode=mode)
                    self.assertEqual(result['status'], status)

    def test_modes_keep_update_initial_state_semantics_distinct(self):
        pair = self.pair([0., 0.], [0., 0.], 'updates')
        for side, before in zip(pair[1:], ([1., 2.], [4., 5.])):
            side['tensors'][0]['shards'] = [dict(offset=0, before=self.array(before),
                after=self.array([value + 1. for value in before]))]
        paths = self.paths(*pair)
        result = finite_compare(*paths)
        self.assertEqual(result['status'], 'incomparable')
        self.assertIn('initial stored parameters differ', result['per_tensor']['weight']['reason'])
        diagnostics = finite_compare(*paths, mode='nonfinite')
        self.assertEqual(diagnostics['global']['paired_finite_count'], 2)
        self.assertEqual(diagnostics['global']['finite_subset_metrics']['rmse']['value'], 0.)
        with self.assertRaisesRegex(ValueError, 'mode'):
            finite_compare(*paths, mode='auto')

    def test_unified_tensor_cli_modes_and_report_equivalence(self):
        paths = self.paths(*self.pair([np.nan, 1., -np.inf], [np.inf, 3., -np.inf]))
        common = ['--contract', str(paths[0]), '--side-a', str(paths[1]), '--side-b', str(paths[2])]
        script = str(SCRIPTS/'compare_tensors.py')
        output = self.root/'unified.json'
        command = [sys.executable, '-I', '-B', script, *common, '--mode', 'nonfinite',
                   '--chunk-size', '1', '--max-locations', '1', '--output', str(output)]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report, compare(*paths, chunk_size=1, max_locations=1))
        saved = output.read_bytes()
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
        self.assertEqual(output.read_bytes(), saved)
        for mode, expected_code in [('metrics', 1), ('invalid', 2)]:
            target = self.root/(mode + '.json')
            run = subprocess.run([sys.executable, '-I', '-B', script, *common, '--mode', mode,
                                  '--output', str(target)], capture_output=True, text=True)
            self.assertEqual(run.returncode, expected_code, run.stderr)
            if mode == 'metrics':
                self.assertEqual(json.loads(target.read_text()), finite_compare(*paths))
            else:
                self.assertFalse(target.exists())
        invalid = self.root/'invalid-limit.json'
        run = subprocess.run([sys.executable, '-I', '-B', script, *common, '--mode', 'nonfinite',
                              '--max-locations', '-1', '--output', str(invalid)], capture_output=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads(invalid.read_text())['status'], 'incomparable')

    def test_multiple_tensor_global_scope_and_sample_bound(self):
        pair = self.pair([np.nan, 2], [np.inf, 3])
        for doc in pair:
            entry = copy.deepcopy(doc['tensors'][0])
            entry['name'] = 'z-last'
            if 'side' in doc:
                entry['shards'] = [dict(offset=0, value=self.array(
                    [np.nan, 2] if doc['side'] == 'a' else [-np.inf, 7]))]
            doc['tensors'].append(entry)
        result = self.run_pair(pair, max_locations=1, chunk_size=1)
        g = result['global']
        self.assertEqual(g['count'], 4)
        self.assertEqual(g['paired_finite_count'], 2)
        self.assertEqual(len(g['samples']), 1)
        self.assertEqual(g['samples'][0]['tensor'], 'weight')
        self.assertEqual(g['finite_subset_metrics']['max_coordinate']['tensor'], 'z-last')
        self.assertEqual(g['finite_subset_metrics']['max_coordinate']['flat_index'], 1)
        self.assertAlmostEqual(g['finite_subset_metrics']['rmse']['value'], np.sqrt(13))
        for tensor in result['per_tensor'].values():
            self.assertEqual(len(tensor['samples']), 1)

    def test_scalar_observation_and_postclip(self):
        pair = self.pair([np.nan], [np.inf], 'observations')
        for doc in pair:
            doc['tensors'][0]['shape'] = []
        result = self.run_pair(pair, chunk_size=1)
        self.assertEqual(result['global']['samples'][0]['coordinate'], [])
        pair = self.pair([1, np.inf], [2, -np.inf], 'postclip-gradients')
        result = self.run_pair(pair)
        self.assertEqual(result['stage'], 'postclip-gradients')
        self.assertEqual(result['global']['joint_counts']['posinf/neginf'], 1)

    def test_truncated_array_is_not_silently_excluded(self):
        pair = self.pair([1, 2], [1, 2])
        descriptor = pair[2]['tensors'][0]['shards'][0]['value']
        path = self.root / descriptor['path']
        path.write_bytes(path.read_bytes()[:-1])
        result = self.run_pair(pair)
        self.assertEqual(result['coverage']['invalid_tensors'], ['weight'])
        self.assertNotEqual(result['per_tensor']['weight']['side_b']['status'], 'no_thresholds')
        self.assertEqual(result['per_tensor']['weight']['side_a']['count'], 2)
        self.assertEqual(result['global']['status'], 'not_covered')

    def test_original_comparator_remains_finite_only(self):
        paths = self.paths(*self.pair([np.nan, 1], [np.inf, 2]))
        self.assertEqual(finite_compare(*paths)['status'], 'incomparable')
        self.assertEqual(compare(*paths)['status'], 'no_thresholds')
        paths = self.paths(*self.pair([1, 3], [2, 7]))
        old, new = finite_compare(*paths), compare(*paths)
        for key in ('rmse', 'cosine', 'max_abs', 'error_norm', 'side_a_norm', 'side_b_norm'):
            self.assertEqual(old['global'][key], new['global']['finite_subset_metrics'][key])


if __name__ == '__main__':
    unittest.main()
