"""Retained bundle replay uses real tools and rejects invalid replay requests."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / 'skills/precision-compare/scripts'
sys.path.insert(0, str(SCRIPTS))
from evidence_bundle import pack
from replay_bundle import check_subset, replay

DEFAULT_DOCUMENT = object()


class ReplayBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='replay-bundle-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.serial = 0
        self.common = dict(model='fixture-model', inputs='fixture-inputs',
                           numerics='float64', scope='fixture', state='initial')
        self.left, self.right = 'z-baseline', 'a-trial'
        self.ranks = {self.left: [0], self.right: [0]}
        for side in self.ranks:
            self.write(side + '-env.json', dict(rank=0, versions={'numpy': np.__version__},
                                               actual=self.common))
            self.write(side + '-status.json', dict(rank=0, status='completed'))
        self.make_artifacts()
        self.case = dict(name='tensor', operation='compare_tensors',
                         inputs=dict(contract='tensors.json', side_a='left-tensors.json',
                                     side_b='right-tensors.json'),
                         expected=dict(status='no_thresholds', sides=dict(side_a=self.left, side_b=self.right)))
        self.case['expected']['global'] = dict(count=2, side_a_norm={'value': 5.},
                                               side_b_norm={'value': 3.}, max_abs={'value': 4.})

    def write(self, name, value):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
        return path

    def base(self, kind, side):
        return dict(schema_version=1, kind=kind, common=copy.deepcopy(self.common), side=side,
                    expected_ranks=[0], ranks=[dict(rank=0, environment=side + '-env.json',
                                                   status=side + '-status.json')])

    def make_artifacts(self):
        tensor = dict(schema_version=1, kind='tensors', common=self.common, thresholds=None,
                      expected_ranks=self.ranks, stage='gradients',
                      tensors=[dict(name='weight', shape=[2], present=True)])
        self.write('tensors.json', tensor)
        distribution = dict(schema_version=1, kind='distributions', common=self.common, thresholds=None,
                            expected_ranks=self.ranks, vocab_size=100, normalization_tolerance=1e-6,
                            renormalization='none', positions=[dict(sample='s', position=0, history=[1])])
        self.write('distributions.json', distribution)
        target = dict(schema_version=1, kind='target-scores', common=self.common, thresholds=None,
                      expected_ranks=self.ranks, targets=[dict(sample='s', position=0, token=1)])
        self.write('targets.json', target)
        for prefix, side, values, logprob in [('left', self.left, [3., 4.], -1.),
                                              ('right', self.right, [3., 0.], -2.)]:
            np.save(self.source / (prefix + '.npy'), np.asarray(values, dtype=np.float64), allow_pickle=False)
            manifest = self.base('tensors', side)
            manifest.update(stage='gradients', tensors=[dict(name='weight', shape=[2], present=True,
                            shards=[dict(offset=0, value={'path': prefix + '.npy'})])])
            self.write(prefix + '-tensors.json', manifest)
            np.save(self.source / (prefix + '-dist.npy'), np.full((1, 100), -np.log(100.)), allow_pickle=False)
            manifest = self.base('distributions', side)
            manifest['positions'] = [dict(sample='s', position=0, history=[1],
                                           scores={'path': prefix + '-dist.npy'})]
            self.write(prefix + '-distributions.json', manifest)
            manifest = self.base('target-scores', side)
            manifest['scores'] = [dict(sample='s', position=0, token=1, logprob=logprob)]
            self.write(prefix + '-targets.json', manifest)

    def bundle(self, document=DEFAULT_DOCUMENT, raw=None, index='EXPORT.json'):
        self.serial += 1
        if raw is None:
            self.write('REPLAY.json', {'cases': [self.case]} if document is DEFAULT_DOCUMENT else document)
        else:
            (self.source / 'REPLAY.json').write_text(raw, encoding='utf-8')
        path = self.root / f'bundle-{self.serial}.zip'
        if index == 'EXPORT.json':
            pack(self.source, path)
        else:
            files = {file.name: file.read_bytes() for file in self.source.iterdir()}
            inventory = {name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
                         for name, data in files.items()}
            with zipfile.ZipFile(path, 'w') as archive:
                for name, data in files.items():
                    archive.writestr(name, data)
                archive.writestr(index, json.dumps(inventory))
        return path

    def output_root(self):
        self.serial += 1
        return self.root / f'output-{self.serial}'

    def cli(self, bundle, entry=None, extra=(), optimize=False):
        return subprocess.run([sys.executable, '-O' if optimize else '-B',
                               str(entry or SCRIPTS / 'replay_bundle.py'),
                               '--bundle', str(bundle), '--output-root', str(self.output_root()), *extra],
                              cwd=self.root, text=True, capture_output=True, encoding='utf-8', check=False)

    def test_real_workflow_all_operations_retains_explicit_side_order(self):
        reverse = copy.deepcopy(self.case)
        reverse['name'] = 'reversed'
        reverse['inputs'].update(side_a='right-tensors.json', side_b='left-tensors.json')
        reverse['expected'].update(sides=dict(side_a=self.right, side_b=self.left))
        reverse['expected']['global'].update(side_a_norm={'value': 3.}, side_b_norm={'value': 5.})
        cases = [self.case, reverse,
                 dict(name='validate', operation='validate',
                      inputs=dict(contract='tensors.json', manifest='left-tensors.json'),
                      expected=dict(status='validated', side=self.left, coordinate_count=2)),
                 dict(name='distributions', operation='compare_distributions',
                      inputs=dict(contract='distributions.json', side_a='left-distributions.json',
                                  side_b='right-distributions.json'),
                      expected=dict(status='no_thresholds', position_count=1,
                                    sides=dict(side_a=self.left, side_b=self.right))),
                 dict(name='targets', operation='compare_target_scores',
                      inputs=dict(contract='targets.json', side_a='left-targets.json', side_b='right-targets.json'),
                      expected=dict(status='no_thresholds', metrics=dict(target_count=1, mae_nat=1.0),
                                    coverage=dict(expected=1, paired=1)))]
        # Extracted Python files are evidence only; installed tools must run.
        (self.source / 'compare_tensors.py').write_text('raise RuntimeError("bundle code executed")\n')
        bundle = self.bundle({'cases': cases, 'scope': 'fixture', 'schema_version': 1})
        root, result = replay(bundle, self.output_root())
        self.assertTrue(result['matched'], result)
        self.assertEqual(result['bundle_sha256'], hashlib.sha256(bundle.read_bytes()).hexdigest())
        self.assertEqual([item['name'] for item in result['cases']], [case['name'] for case in cases])
        first = json.loads((root / 'case-00.json').read_text())
        reversed_report = json.loads((root / 'case-01.json').read_text())
        self.assertEqual(first['sides'], dict(side_a=self.left, side_b=self.right))
        self.assertEqual(reversed_report['sides'], dict(side_a=self.right, side_b=self.left))
        self.assertEqual(json.loads((root / 'replay.json').read_text()), result)

    def test_nonfinite_diagnostics_replay_retains_anomalies_and_finite_scope(self):
        np.save(self.source / 'left.npy', np.array([np.nan, 4.]), allow_pickle=False)
        np.save(self.source / 'right.npy', np.array([np.inf, 2.]), allow_pickle=False)
        case = copy.deepcopy(self.case)
        case.update(name='nonfinite', operation='inspect_nonfinite', expected=dict(
            kind='nonfinite_tensors', status='no_thresholds'))
        case['expected']['global'] = dict(
            count=2, paired_finite_count=1, paired_finite_fraction=0.5,
            side_a=dict(nan=1, finite=1), side_b=dict(posinf=1, finite=1),
            joint_counts={'nan/posinf': 1, 'finite/finite': 1},
            finite_subset_metrics=dict(count=1, max_abs={'value': 2.},
                                      max_coordinate=dict(tensor='weight', flat_index=1,
                                                          coordinate=[1])))
        # Retained code must not replace the installed inspection tool.
        (self.source / 'inspect_nonfinite.py').write_text('raise RuntimeError("bundle code executed")\n')
        bundle = self.bundle({'cases': [case]})
        root, result = replay(bundle, self.output_root())
        self.assertTrue(result['matched'], result)
        report = json.loads((root / 'case-00.json').read_text())
        self.assertIsNone(report['thresholds'])
        self.assertEqual(report['global']['samples'][0]['flat_index'], 0)

    def test_nonfinite_replay_checks_retained_array_references(self):
        case = copy.deepcopy(self.case)
        case['operation'] = 'inspect_nonfinite'
        manifest = json.loads((self.source / 'left-tensors.json').read_text())
        manifest['tensors'][0]['shards'][0]['value']['path'] = '../outside.npy'
        self.write('left-tensors.json', manifest)
        bundle = self.bundle({'cases': [case]})
        with self.assertRaisesRegex(ValueError, 'leaves retained inputs'):
            replay(bundle, self.output_root())

    def test_wrong_metric_is_recorded_and_cli_returns_nonzero(self):
        self.case['expected']['global']['max_abs']['value'] = 9.
        bundle = self.bundle()
        root, result = replay(bundle, self.output_root())
        self.assertFalse(result['matched'])
        self.assertEqual(result['cases'][0]['assertion'], 'failed')
        self.assertIn('global.max_abs.value', result['cases'][0]['reason'])
        self.assertTrue((root / 'case-00.json').is_file())
        completed = self.cli(bundle)
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertFalse(json.loads(completed.stdout)['matched'])

    def test_tampered_payload_rejected_also_under_optimized_python(self):
        bundle = self.bundle()
        with zipfile.ZipFile(bundle) as archive:
            members = {info.filename: archive.read(info) for info in archive.infolist()}
        members['left-tensors.json'] += b' '
        tampered = self.root / 'tampered.zip'
        with zipfile.ZipFile(tampered, 'w') as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        output = self.output_root()
        with self.assertRaises(ValueError):
            replay(tampered, output)
        self.assertEqual(list(output.glob('*/inputs')), [])
        completed = self.cli(tampered, optimize=True)
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, '')
        self.assertTrue(completed.stderr)

    def test_alternate_files_index_and_default_cli_from_other_cwd(self):
        bundle = self.bundle(index='FILES.json')
        completed = self.cli(bundle, extra=('--index', 'FILES.json', '--scripts', str(SCRIPTS)))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(json.loads(completed.stdout)['matched'])
        # Both supported index layouts use the installed tool from another cwd.
        completed = self.cli(self.bundle())
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(json.loads(completed.stdout)['matched'])

    def test_explicit_other_checkout_tools_are_selected(self):
        selected = self.root / 'other-checkout' / 'scripts'
        selected.mkdir(parents=True)
        for name in ('compare_target_scores.py', 'conditions.py', 'score_metrics.py'):
            shutil.copyfile(SCRIPTS / name, selected / name)
        # Add a version marker only in the explicitly selected copy.
        tool = selected / 'compare_target_scores.py'
        text = tool.read_text(encoding='utf-8')
        text = text.replace("kind='target-scores', schema_version=1", "selected_version='other-checkout', kind='target-scores', schema_version=1", 1)
        tool.write_text(text, encoding='utf-8')
        case = dict(name='selected', operation='compare_target_scores',
                    inputs=dict(contract='targets.json', side_a='left-targets.json', side_b='right-targets.json'),
                    expected=dict(status='no_thresholds', selected_version='other-checkout'))
        completed = self.cli(self.bundle({'cases': [case]}), extra=('--scripts', str(selected)))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(json.loads(completed.stdout)['matched'])

    def test_external_array_absent_from_bundle_cannot_match(self):
        external = self.root / 'outside-bundle.npy'
        np.save(external, np.asarray([3., 4.]), allow_pickle=False)
        manifest = json.loads((self.source / 'left-tensors.json').read_text())
        manifest['tensors'][0]['shards'][0]['value']['path'] = str(external)
        self.write('left-tensors.json', manifest)
        (self.source / 'left.npy').unlink()
        with self.assertRaises(ValueError):
            replay(self.bundle(), self.output_root())

    def test_external_loaded_references_rejected_before_any_case_runs(self):
        external_array = self.root / 'outside-bundle.npy'
        np.save(external_array, np.asarray([3., 4.]), allow_pickle=False)
        external_json = self.root / 'outside-bundle.json'
        external_json.write_text(json.dumps(dict(rank=0, status='completed', versions={'numpy': np.__version__},
                                                actual=self.common)), encoding='utf-8')
        for operation, field in [('compare_tensors', 'value'), ('compare_tensors', 'before'),
                                 ('compare_tensors', 'after'), ('compare_distributions', 'scores'),
                                 ('compare_tensors', 'environment'), ('compare_tensors', 'status'),
                                 ('compare_distributions', 'environment'), ('compare_distributions', 'status'),
                                 ('compare_target_scores', 'environment'), ('compare_target_scores', 'status'),
                                 ('validate', 'environment'), ('validate', 'status')]:
            for absolute in (False, True):
                with self.subTest(operation=operation, field=field, absolute=absolute):
                    self.make_artifacts()
                    kind = {'compare_tensors': 'tensors', 'validate': 'tensors',
                            'compare_distributions': 'distributions', 'compare_target_scores': 'targets'}[operation]
                    case = dict(name='unsafe', operation=operation,
                                inputs=dict(contract={'tensors': 'tensors.json', 'distributions': 'distributions.json',
                                                      'targets': 'targets.json'}[kind]),
                                expected={'status': 'no_thresholds'})
                    if operation == 'validate':
                        case['inputs']['manifest'] = 'left-' + kind + '.json'
                    else:
                        case['inputs'].update(side_a='left-' + kind + '.json', side_b='right-' + kind + '.json')
                    if field in ('before', 'after'):
                        contract = json.loads((self.source / 'tensors.json').read_text())
                        contract['stage'] = 'updates'
                        self.write('tensors.json', contract)
                        for prefix in ('left', 'right'):
                            manifest = json.loads((self.source / (prefix + '-tensors.json')).read_text())
                            manifest.update(stage='updates', events={'actor': 'applied'})
                            manifest['tensors'][0]['shards'] = [dict(offset=0, before={'path': prefix + '.npy'},
                                                                   after={'path': prefix + '.npy'})]
                            self.write(prefix + '-tensors.json', manifest)
                    path = external_json if field in ('environment', 'status') else external_array
                    reference = str(path) if absolute else '../../../' + path.name
                    manifest = json.loads((self.source / ('left-' + kind + '.json')).read_text())
                    if field in ('environment', 'status'):
                        manifest['ranks'][0][field] = reference
                    elif field == 'scores':
                        manifest['positions'][0]['scores']['path'] = reference
                    else:
                        manifest['tensors'][0]['shards'][0][field]['path'] = reference
                    self.write('left-' + kind + '.json', manifest)
                    # Even an earlier valid case must wait for all reference checks.
                    bundle = self.bundle({'cases': [dict(name='earlier', operation='validate',
                                                        inputs=dict(contract='tensors.json', manifest='right-tensors.json'),
                                                        expected={'status': 'validated'}), case]})
                    with patch('replay_bundle.subprocess.run', side_effect=AssertionError('tool ran before closure check')) as run:
                        with self.assertRaises(ValueError):
                            replay(bundle, self.output_root())
                        run.assert_not_called()
        completed = self.cli(bundle, optimize=True)
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, '')

    def test_contained_parent_references_and_provenance_labels_remain_portable(self):
        for prefix in ('left', 'right'):
            manifest = json.loads((self.source / (prefix + '-tensors.json')).read_text())
            for rank in manifest['ranks']:
                for field in ('environment', 'status'):
                    rank[field] = '../' + rank[field]
            manifest['tensors'][0]['shards'][0]['value']['path'] = '../' + prefix + '.npy'
            manifest['provenance'] = {'path': str(self.root / 'unretained-source.py')}
            self.write('captures/' + prefix + '-tensors.json', manifest)
            self.case['inputs']['side_a' if prefix == 'left' else 'side_b'] = 'captures/' + prefix + '-tensors.json'
        _, result = replay(self.bundle(), self.output_root())
        self.assertTrue(result['matched'], result)

    def test_replay_expansion_limit_api_and_cli(self):
        bundle = self.bundle()
        with patch('replay_bundle.subprocess.run', side_effect=AssertionError('tool ran before expansion check')) as run:
            with self.assertRaises(ValueError):
                replay(bundle, self.output_root(), max_expanded_bytes=1)
            run.assert_not_called()
        completed = self.cli(bundle, extra=('--max-expanded-bytes', '1'))
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, '')
        self.assertIn('expanded', completed.stderr)
        completed = self.cli(bundle, extra=('--max-expanded-bytes', str(1024 * 1024)))
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_invalid_cases_rejected_before_tool_execution(self):
        invalid = [None, [], {}, {'cases': []}, {'cases': {}}, {'cases': [None]},
                   {'cases': [self.case, self.case]}]
        for field, values in [('name', ['', ' ', 1]),
                              ('operation', ['unknown', ['validate'], None]),
                              ('inputs', [None, [], {}, {'contract': 'tensors.json'}]),
                              ('expected', [None, [], {}, 1, 'status'])]:
            for value in values:
                case = copy.deepcopy(self.case)
                case[field] = value
                invalid.append({'cases': [case]})
        for path in ('../tensors.json', '/tensors.json', 'C:/tensors.json', 'a\\b.json',
                     './tensors.json', 'missing.json', '.', 'tensors.json/', 1, None):
            case = copy.deepcopy(self.case)
            case['inputs']['contract'] = path
            invalid.append({'cases': [case]})
        case = copy.deepcopy(self.case)
        case['inputs']['extra'] = 'tensors.json'
        invalid.append({'cases': [case]})
        case = copy.deepcopy(self.case)
        del case['expected']
        invalid.append({'cases': [case]})
        for document in invalid:
            with self.subTest(document=document):
                bundle = self.bundle(document)
                with patch('replay_bundle.subprocess.run') as run:
                    with self.assertRaises(ValueError):
                        replay(bundle, self.output_root())
                    run.assert_not_called()

    def test_duplicate_keys_and_nonfinite_expected_rejected_under_optimization(self):
        raw = json.dumps({'cases': [self.case]})
        invalid = [raw.replace('"name": "tensor"', '"name": "tensor", "name": "duplicate"'),
                   raw.replace('"operation": "compare_tensors"', '"operation": "validate", "operation": "compare_tensors"'),
                   raw.replace('"contract": "tensors.json"', '"contract": "tensors.json", "contract": "tensors.json"'),
                   raw.replace('"status": "no_thresholds"', '"status": "no_thresholds", "status": "no_thresholds"')]
        for token in ('NaN', 'Infinity', '-Infinity', '1e999'):
            invalid.append(raw.replace('"value": 4.0', '"value": ' + token))
        for text in invalid:
            with self.subTest(raw=text):
                bundle = self.bundle(raw=text)
                with patch('replay_bundle.subprocess.run') as run:
                    with self.assertRaises(ValueError):
                        replay(bundle, self.output_root())
                    run.assert_not_called()
        completed = self.cli(self.bundle(raw=invalid[-1]), optimize=True)
        self.assertEqual(completed.returncode, 1)
        self.assertIn('nonfinite', completed.stderr)

    def test_nested_subset_numeric_tolerance_types_and_list_order(self):
        check_subset({'x': [{'value': 1. + 5e-13, 'extra': True}]}, {'x': [{'value': 1.}]})
        for actual, expected in [({'x': 1. + 1e-8}, {'x': 1.}), ({'x': True}, {'x': 1}),
                                 ({'x': 1.}, {'x': 1}), ({'x': [1, 2]}, {'x': [2, 1]}),
                                 ({'x': [1]}, {'x': [1, 2]}), ({'x': float('inf')}, {'x': 1.})]:
            with self.subTest(actual=actual, expected=expected):
                with self.assertRaises(AssertionError):
                    check_subset(actual, expected)


if __name__ == '__main__':
    unittest.main()
