"""Numerical and malformed-artifact regression tests for portable offline tools."""
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills/precision-skill/scripts'
sys.path.insert(0, str(SCRIPTS))
from artifact_io import Array, ArtifactError, load_json
from compare_tensors import Stats, compare as tensors
from compare_distributions import compare as distributions
from validate_artifacts import validate


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.serial = 0
        for side in ['GPU', 'NPU']:
            self.write(f'{side}-env.json', {'rank': 0, 'versions': {'numpy': np.__version__}})
            self.write(f'{side}-status.json', {'rank': 0, 'status': 'completed'})
        self.common = {'model': 'fixture-model-v1', 'inputs': 'fixture-inputs-v1', 'numerics': 'float64', 'scope': 'fixture', 'state': 'initial-state-v1'}

    def write(self, name, obj):
        path = self.root/name
        path.write_text(json.dumps(obj, allow_nan=False))
        return path

    def array(self, data, dtype=np.float64):
        self.serial += 1
        path = self.root/f'array{self.serial}.npy'
        np.save(path, np.asarray(data, dtype=dtype), allow_pickle=False)
        return {'path': path.name}

    def base(self, kind, side):
        return {'schema_version': 1, 'kind': kind, 'side': side, 'common': copy.deepcopy(self.common),
                'expected_ranks': [0], 'ranks': [{'rank': 0, 'environment': f'{side}-env.json', 'status': f'{side}-status.json'}]}

    def tensor_pair(self, r=(3., 4.), c=(3., 0.), stage='gradients'):
        spec = {'schema_version': 1, 'kind': 'tensors', 'common': self.common, 'thresholds': None, 'expected_ranks': {'GPU': [0], 'NPU': [0]}, 'stage': stage,
                'tensors': [{'name': 'weight', 'shape': [len(r)], 'present': True}]}
        manifests = []
        for side, values in [('GPU', r), ('NPU', c)]:
            m = self.base('tensors', side)
            m.update(stage=stage, events={'actor': 'applied'}, tensors=[{'name': 'weight', 'shape': [len(values)], 'present': True,
                            'shards': [{'offset': 0, 'value': self.array(values)}]}])
            manifests.append(m)
        return spec, *manifests

    def run_tensors(self, spec, r, c, chunk=1):
        return tensors(self.write('contract.json', spec), self.write('gpu.json', r), self.write('npu.json', c), chunk)

    def dist_pair(self, p=None, q=None):
        if p is None:
            p = np.arange(202, 0, -1, dtype=np.float64)
            p /= p.sum()
        if q is None:
            q = p.copy()
            q[[0, 100]] = q[[100, 0]]
        with np.errstate(divide='ignore'):
            lp, lq = np.log(p), np.log(q)
        spec = {'schema_version': 1, 'kind': 'distributions', 'common': self.common, 'thresholds': None, 'expected_ranks': {'GPU': [0], 'NPU': [0]},
                'vocab_size': len(p), 'normalization_tolerance': 1e-6, 'renormalization': 'none', 'positions': [{'sample': 'a', 'position': 0, 'history': [1]}]}
        manifests = []
        for side, values in [('GPU', lp), ('NPU', lq)]:
            m = self.base('distributions', side)
            m['positions'] = [{**copy.deepcopy(spec['positions'][0]), 'scores': self.array([values])}]
            manifests.append(m)
        return spec, *manifests

    def run_dist(self, spec, r, c):
        return distributions(self.write('contract.json', spec), self.write('gpu.json', r), self.write('npu.json', c))

    def test_tensor_hand_values_and_chunking(self):
        spec, r, c = self.tensor_pair()
        report = self.run_tensors(spec, r, c)
        result = report['global']
        self.assertEqual(report['status'], 'no_thresholds')
        self.assertAlmostEqual(result['relative_l2']['value'], .8)
        self.assertAlmostEqual(result['cosine']['value'], .6)
        self.assertEqual(result['max_abs']['value'], 4)
        self.assertEqual(result['max_flat_index'], 1)
        self.assertEqual(report, self.run_tensors(spec, r, c, chunk=100))

    def test_global_is_concatenation_not_ratio_average(self):
        spec, r, c = self.tensor_pair((100.,), (0.,))
        spec['tensors'].append({'name': 'small', 'shape': [1], 'present': True})
        for m, value in [(r, 1.), (c, 2.)]:
            m['tensors'].append({'name': 'small', 'shape': [1], 'present': True, 'shards': [{'offset': 0, 'value': self.array([value])}]})
        result = self.run_tensors(spec, r, c)
        self.assertAlmostEqual(result['global']['relative_l2']['value'], 1.)
        self.assertEqual(result['global']['count'], 2)

    def test_independent_shard_layouts(self):
        spec, r, c = self.tensor_pair((1, 2, 3, 4), (2, 3, 4, 5))
        r['tensors'][0]['shards'] = [{'offset': 0, 'value': self.array([1, 2])}, {'offset': 2, 'value': self.array([3, 4])}]
        c['tensors'][0]['shards'] = [{'offset': 0, 'value': self.array([2])}, {'offset': 1, 'value': self.array([3, 4, 5])}]
        result = self.run_tensors(spec, r, c, chunk=3)
        self.assertAlmostEqual(result['global']['relative_l2']['value'], 2/math.sqrt(30))

    def test_update_delta_and_initial_state(self):
        spec, r, c = self.tensor_pair((0., 0.), (0., 0.), stage='updates')
        for m, after in [(r, [1., 2.]), (c, [1.125, 2.])]:
            m['tensors'][0]['shards'] = [{'offset': 0, 'before': self.array([1., 2.], np.float16), 'after': self.array(after, np.float16)}]
        c['events']['actor'] = 'skipped'
        result = self.run_tensors(spec, r, c)
        self.assertIsNone(result['global']['relative_l2']['value'])
        self.assertEqual(result['global']['max_abs']['value'], .125)
        self.assertEqual(result['per_tensor']['weight']['initial_values'], 'exactly_checked')
        c['tensors'][0]['shards'][0]['before'] = self.array([1.5, 2.])
        self.assertEqual(self.run_tensors(spec, r, c)['status'], 'incomparable')

    def test_bf16_reader_and_promoted_difference(self):
        # Exact stored BF16 1 and 2^-9; their difference is not BF16 representable.
        header = json.dumps({'x': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]}}).encode()
        (self.root/'a.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + struct.pack('<HH', 0x3f80, 0x3b00))
        arr = Array(self.root, {'path': 'a.safetensors', 'key': 'x'})
        np.testing.assert_array_equal(arr.read(0, 2), [1., 2**-9])
        self.assertEqual(arr.read(0, 1)[0]-arr.read(1, 1)[0], .998046875)

    def test_reject_bf16_or_fp16_stored_delta(self):
        spec, r, c = self.tensor_pair(stage='updates')
        c['tensors'][0]['shards'][0]['value'] = self.array([3., 0.], np.float16)
        self.assertEqual(self.run_tensors(spec, r, c)['status'], 'incomparable')

    def test_zero_norms(self):
        for a, b, relative in [([0.], [0.], 0.), ([0.], [1.], None), ([1.], [0.], 1.)]:
            result = self.run_tensors(*self.tensor_pair(a, b))['global']
            self.assertEqual(result['relative_l2']['value'], relative)
            self.assertIsNone(result['cosine']['value'])
            self.assertEqual(result['max_abs']['status'], 'no_thresholds')

    def test_extreme_norms_stay_stable(self):
        report = self.run_tensors(*self.tensor_pair([1e300, 1e300], [1e300, 0.]))
        self.assertAlmostEqual(report['global']['relative_l2']['value'], 1/math.sqrt(2))
        self.assertAlmostEqual(report['global']['cosine']['value'], 1/math.sqrt(2))

    def test_absence_is_not_zero(self):
        spec, r, c = self.tensor_pair()
        c['tensors'][0].update(present=False, shards=[])
        self.assertEqual(self.run_tensors(spec, r, c)['status'], 'incomparable')
        spec['tensors'][0]['present'] = False
        r['tensors'][0].update(present=False, shards=[])
        result = self.run_tensors(spec, r, c)
        self.assertEqual(result['per_tensor']['weight']['presence'], 'absent_on_both_sides')
        self.assertEqual(result['global']['count'], 0)

    def test_invalid_tensor_inventory_cases(self):
        mutations = [lambda x: x['tensors'].clear(), lambda x: x['tensors'].append(copy.deepcopy(x['tensors'][0])),
                     lambda x: x['tensors'][0]['shards'][0].update(offset=1),
                     lambda x: x['tensors'][0]['shards'].append(copy.deepcopy(x['tensors'][0]['shards'][0])),
                     lambda x: x['tensors'][0].update(shape=[1]),
                     lambda x: x['tensors'][0]['shards'][0].update(value=self.array([float('nan'), 1.])),
                     lambda x: x['tensors'][0]['shards'][0].update(value={'path': 'absent.npy'})]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                spec, r, c = self.tensor_pair()
                mutate(c)
                result = self.run_tensors(spec, r, c)
                self.assertNotEqual(result['status'], 'no_thresholds')
                self.assertNotIn('relative_l2', result.get('global', {}))

    def test_valid_other_tensor_preserved(self):
        spec, r, c = self.tensor_pair()
        spec['tensors'].append({'name': 'bad', 'shape': [1], 'present': True})
        for m in [r, c]:
            m['tensors'].append({'name': 'bad', 'shape': [1], 'present': True, 'shards': []})
        result = self.run_tensors(spec, r, c)
        self.assertIn('relative_l2', result['per_tensor']['weight'])
        self.assertNotIn('relative_l2', result['global'])

    def test_distribution_hand_kl_union(self):
        spec, r, c = self.dist_pair()
        result = self.run_dist(spec, r, c)
        row = result['positions'][0]
        p0, p100 = 202/(202*203/2), 102/(202*203/2)
        expected = (p0-p100)*math.log(p0/p100)
        self.assertEqual(row['union_count'], 101)
        self.assertAlmostEqual(row['tail_merged_kl']['value'], expected, places=14)
        self.assertAlmostEqual(row['probability_rmse'], math.sqrt(2*(p0-p100)**2/101))
        self.assertAlmostEqual(row['top100_overlap'], .99)
        self.assertFalse(row['top1_agreement'])
        self.assertAlmostEqual(row['top1_mismatch']['reference_margin'], 1/(202*203/2))

    def test_distribution_true_zero_keeps_other_metrics(self):
        p = np.ones(202)/202
        q = p.copy();q[0] = 0;q[201] += p[0]
        result = self.run_dist(*self.dist_pair(p, q))
        self.assertEqual(result['status'], 'incomparable')
        self.assertIsNone(result['aggregate']['tail_merged_kl']['value'])
        self.assertIn('mean', result['aggregate']['probability_rmse'])
        self.assertIn('at 0', result['positions'][0]['tail_merged_kl']['reason'])

    def test_log_tail_underflow_is_not_true_zero(self):
        spec, r, c = self.dist_pair(np.ones(128)/128, np.ones(128)/128)
        lp = np.r_[np.full(100, -math.log(100)), np.full(28, -1000.)]
        lq = np.r_[np.full(100, -math.log(100)), np.full(28, -math.inf)]
        r['positions'][0]['scores'] = self.array(lp)
        c['positions'][0]['scores'] = self.array(lq)
        result = self.run_dist(spec, r, c)
        self.assertIn('other', result['positions'][0]['tail_merged_kl']['reason'])

    def test_ties_resolve_by_token_id(self):
        spec, r, c = self.dist_pair(np.ones(128)/128, np.ones(128)/128)
        result = self.run_dist(spec, r, c)['positions'][0]
        self.assertEqual(result['union_count'], 100)
        self.assertEqual(result['max_error_token'], 0)
        self.assertEqual(result['tail_merged_kl']['value'], 0.)

    def test_position_aggregation_not_averaged_percentiles(self):
        spec, r, c = self.dist_pair()
        spec['positions'].append({'sample': 'b', 'position': 1, 'history': [1, 2]})
        for m in [r, c]:
            m['positions'].append({**spec['positions'][1], 'scores': copy.deepcopy(r['positions'][0]['scores'])})
        c['positions'].reverse()
        result = self.run_dist(spec, r, c)
        first = result['positions'][0]['max_probability_error']
        self.assertAlmostEqual(result['aggregate']['max_probability_error']['mean'], first/2)
        self.assertAlmostEqual(result['aggregate']['max_probability_error']['p99'], first*.99)
        self.assertEqual(result['aggregate']['top1_agreement']['count'], 1)

    def test_invalid_distribution_cases(self):
        mutations = [lambda x: x['positions'].clear(), lambda x: x['positions'].append(copy.deepcopy(x['positions'][0])),
                     lambda x: x['positions'][0].update(history=[2]),
                     lambda x: x['positions'][0]['scores'].update(row=99),
                     lambda x: x['positions'][0].update(scores=self.array(np.full(100, -math.log(100)))),
                     lambda x: x['positions'][0].update(scores=self.array(np.full(202, -math.log(202)+.01))),
                     lambda x: x['positions'][0].update(scores=self.array(np.full(202, np.nan)))]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                spec, r, c = self.dist_pair();mutate(c)
                result = self.run_dist(spec, r, c)
                self.assertNotEqual(result['status'], 'no_thresholds')
                self.assertNotIn('probability_rmse', result.get('aggregate', {}))

    def test_common_and_rank_evidence(self):
        for mutation in [lambda x: x['common'].update(model='wrong'), lambda x: x['ranks'].clear(),
                         lambda x: x.update(expected_ranks=[0, 1]), lambda x: x.update(side='GPU')]:
            spec, r, c = self.tensor_pair();mutation(c)
            self.assertNotEqual(self.run_tensors(spec, r, c)['status'], 'no_thresholds')
        self.write('NPU-status.json', {'rank': 0, 'status': 'failed'})
        self.assertEqual(self.run_tensors(*self.tensor_pair())['status'], 'execution_failed')

    def test_single_side_validation(self):
        for build in [self.tensor_pair, self.dist_pair]:
            spec, _, cand = build()
            result = validate(self.write('contract.json', spec), self.write('npu.json', cand), chunk_size=1)
            self.assertEqual(result['status'], 'validated')
            self.assertEqual(result['side'], 'NPU')
        spec, _, cand = self.tensor_pair()
        cand['tensors'][0]['shards'][0]['value'] = self.array([math.inf, 1.])
        self.assertEqual(validate(self.write('contract.json', spec), self.write('npu.json', cand))['status'], 'incomparable')

    def test_self_check_and_cli_exclusive_output(self):
        spec, ref, _ = self.tensor_pair()
        cp, rp = self.write('contract.json', spec), self.write('gpu.json', ref)
        self.assertTrue(tensors(cp, rp, rp)['self_comparison'])
        out = self.root/'cli.json'
        cmd = [sys.executable, str(SCRIPTS/'compare_tensors.py'), '--contract', str(cp), '--reference', str(rp), '--candidate', str(rp), '--output', str(out)]
        self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
        contents = out.read_bytes()
        self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 2)
        self.assertEqual(out.read_bytes(), contents)

    def test_rank_inventory_cannot_be_shrunk_to_hide_missing_rank(self):
        spec, r, c = self.tensor_pair()
        spec['expected_ranks']['NPU'] = [0, 1]
        result = self.run_tensors(spec, r, c)
        self.assertEqual(result['status'], 'not_covered')

    def test_explicit_renormalization_matches_original_mass(self):
        spec, r, c = self.dist_pair(np.ones(128)/128, np.ones(128)/128)
        spec['renormalization'] = 'full_vocabulary_fp64'
        c['positions'][0]['scores'] = self.array(np.full(128, -math.log(128)+1e-7))
        result = self.run_dist(spec, r, c)
        self.assertEqual(result['status'], 'no_thresholds')
        self.assertAlmostEqual(result['positions'][0]['normalization_mass']['candidate'], math.exp(1e-7))
        self.assertLess(result['aggregate']['probability_rmse']['max'], 1e-15)

    def test_invalid_position_blocks_aggregate_not_valid_position(self):
        spec, r, c = self.dist_pair()
        spec['positions'].append({'sample': 'b', 'position': 0, 'history': [1]})
        for m in [r, c]:
            m['positions'].append({**spec['positions'][1], 'scores': self.array(np.full(202, np.nan))})
        result = self.run_dist(spec, r, c)
        self.assertEqual(result['status'], 'incomparable')
        self.assertIn('probability_rmse', result['positions'][0])
        self.assertNotIn('probability_rmse', result['aggregate'])

    def test_no_threshold_fitting(self):
        spec, r, c = self.tensor_pair()
        spec['thresholds'] = {'relative_l2': 99}
        self.assertEqual(self.run_tensors(spec, r, c)['status'], 'incomparable')

    def test_missing_file_status_and_blocked_cli(self):
        spec, r, c = self.tensor_pair()
        c['tensors'][0]['shards'][0]['value'] = {'path': 'missing.npy'}
        result = self.run_tensors(spec, r, c)
        self.assertEqual(result['status'], 'not_covered')
        out = self.root/'blocked.json'
        proc = subprocess.run([sys.executable, str(SCRIPTS/'compare_tensors.py'),
            '--contract', str(self.root/'contract.json'), '--reference', str(self.root/'gpu.json'),
            '--candidate', str(self.root/'npu.json'), '--output', str(out)], capture_output=True)
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(json.loads(out.read_text())['status'], 'not_covered')

    def test_inactive_schedule_difference_is_not_numerical_skip(self):
        spec, r, c = self.tensor_pair(stage='updates')
        c['events']['actor'] = 'inactive'
        self.assertEqual(self.run_tensors(spec, r, c)['status'], 'incomparable')
        c['events']['actor'] = 'skipped'
        self.assertEqual(self.run_tensors(spec, r, c)['status'], 'no_thresholds')

    def test_scalar_parameter(self):
        spec, r, c = self.tensor_pair((1.,), (2.,))
        spec['tensors'][0]['shape'] = []
        for m in [r, c]:m['tensors'][0]['shape'] = []
        self.assertEqual(self.run_tensors(spec, r, c)['global']['count'], 1)

    def test_distribution_and_audit_cli(self):
        spec, ref, cand = self.dist_pair()
        cp, rp, np_ = self.write('contract.json', spec), self.write('gpu.json', ref), self.write('npu.json', cand)
        for entry, args in [('compare_distributions.py', ['--reference', str(rp), '--candidate', str(np_)]),
                            ('validate_artifacts.py', ['--manifest', str(rp)])]:
            out = self.root/(entry+'.json')
            proc = subprocess.run([sys.executable, str(SCRIPTS/entry), '--contract', str(cp), *args, '--output', str(out)], capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
            self.assertTrue(out.exists())

    def test_selected_union_queries_match_full_array_metrics(self):
        from compare_distributions import per_position, selected_metrics, lse
        for size in (100, 200, 257):
            p = np.arange(1, size + 1, dtype=np.float64)
            q = np.roll(p**2, 53)
            lp, lq = np.log(p/p.sum()), np.log(q/q.sum())
            ids = np.arange(size)
            tg, tn = np.lexsort((ids, -lp))[:100], np.lexsort((ids, -lq))[:100]
            union = np.union1d(tg, tn)
            tail = ~np.isin(ids, union)
            pl, ql = np.append(lp[union], lse(lp[tail])), np.append(lq[union], lse(lq[tail]))
            queried = selected_metrics(union, pl, ql, tg, tn, normalization_tolerance=1e-12)
            self.assertEqual(queried, per_position(lp, lq))
            with self.assertRaises(ArtifactError):
                selected_metrics(union[:-1], pl, ql, tg, tn)
            with self.assertRaises(ArtifactError):
                selected_metrics(union, pl - .1, ql, tg, tn, normalization_tolerance=1e-12)

    def test_unrenormalized_scores_keep_original_mass_tolerance(self):
        spec, r, c = self.dist_pair()
        # Scores within the declared serialization tolerance remain valid without renormalization.
        for manifest in (r, c):
            descriptor = manifest['positions'][0]['scores']
            values = np.load(self.root / descriptor['path']) - 1e-7
            manifest['positions'][0]['scores'] = self.array(values)
        self.assertEqual(self.run_dist(spec, r, c)['status'], 'no_thresholds')

    def test_strict_json(self):
        p = self.root/'bad.json'
        for s in ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}']:
            p.write_text(s)
            with self.assertRaises(ArtifactError):load_json(p)

    def test_truncated_safetensors_and_unsafe_numpy(self):
        (self.root/'bad.safetensors').write_bytes(struct.pack('<Q', 999))
        with self.assertRaises(ArtifactError):Array(self.root, {'path': 'bad.safetensors', 'key': 'x'})
        np.save(self.root/'object.npy', np.array([{'x': 1}], dtype=object))
        with self.assertRaises(ValueError):Array(self.root, {'path': 'object.npy'})


if __name__ == '__main__':
    unittest.main()
