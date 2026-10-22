"""Logical history checks, artifact coordinates and native capture boundaries."""
import importlib.util
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

TOOLS = Path(__file__).resolve().parents[1]/'skills/precision-compare/scripts'
sys.path.insert(0, str(TOOLS))
from inference_capture import (FixedHistory, derive_logprobs, export_records,
                               validate_generation_scores, validate_history)


class InferenceExportTests(unittest.TestCase):
    def test_positions_target_shift_zero_probability_and_duplicates(self):
        values = derive_logprobs(np.arange(5, dtype=np.float32))
        record = dict(sample='request', position=2, history=[1, 2, 3], token=4,
                      logprob=float(values[4]), scores=dict(path='row.npy', row=0))
        with tempfile.TemporaryDirectory() as root:
            np.save(Path(root)/'row.npy', values)
            result = export_records([record], root)
            self.assertEqual(result['positions'][0]['position'], 2)
            self.assertEqual(result['scores'][0]['position'], 3)
            self.assertEqual(result['positions'][0]['scores']['path'], str((Path(root)/'row.npy').resolve()))
            zeros = np.full(5, -np.log(4.)); zeros[4] = -np.inf
            np.save(Path(root)/'zero.npy', zeros)
            zero = export_records([{**record, 'logprob': None,
                                    'scores':dict(path='zero.npy', row=0)}], root)
            self.assertEqual(zero['scores'], [])
            self.assertEqual(zero['unavailable_targets'][0]['position'], 3)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                export_records([record, record], root)
            with self.assertRaisesRegex(ValueError, 'history'):
                export_records([{**record, 'history': [1]}], root)

    def test_native_histories_and_mrope_coordinates(self):
        events = [dict(stage='prefill', input_ids=[1, 2], positions=[[0, 1]]*3, prefix_length=0),
                  dict(stage='decode', input_ids=[3], positions=[[2]]*3)]
        self.assertEqual(validate_history([1, 2], [3, 4], events)['decode_tokens'], 1)
        events[1]['input_ids'] = [4]
        with self.assertRaisesRegex(ValueError, 'step 1'):
            validate_history([1, 2], [3, 4], events)
        events[1]['input_ids'] = [3]
        events[0]['prefix_length'] = 1
        with self.assertRaisesRegex(ValueError, 'step 0'):
            validate_history([1, 2], [3, 4], events)

    def test_integer_coordinates_and_decode_prefix_cannot_be_coerced(self):
        original = [dict(stage='prefill', input_ids=[1, 2], positions=[0, 1], prefix_length=0),
                    dict(stage='decode', input_ids=[3], positions=[2], prefix_length=2)]
        self.assertEqual(validate_history([1, 2], [3, 4], original)['distributions'], 2)
        for step, key, value in [(0,'input_ids',[True,2]), (0,'input_ids',[1,2.0]),
                                 (0,'positions',[False,True]), (0,'positions',[[0.,1.]]),
                                 (0,'prefix_length',False), (1,'input_ids',[3.]),
                                 (1,'positions',[2.]), (1,'prefix_length',999),
                                 (1,'prefix_length',2.0)]:
            events = copy.deepcopy(original); events[step][key] = value
            with self.subTest(step=step,key=key,value=value), self.assertRaises(ValueError):
                validate_history([1,2], [3,4], events)
        # Existing adapters may omit decode prefix_length; supplied values are checked.
        del original[1]['prefix_length']
        self.assertEqual(validate_history([1,2], [3,4], original)['decode_tokens'],1)

    def test_export_checks_actual_row_and_target_value(self):
        with tempfile.TemporaryDirectory() as root:
            values = derive_logprobs(np.array([[0.,1.,2.,3.,4.],[4.,3.,2.,1.,0.]],dtype=np.float32))
            np.save(Path(root)/'rows.npy',values)
            record = dict(sample='r', position=1, history=[1,2], token=4,
                          logprob=float(values[1,4]), scores=dict(path='rows.npy',row=1))
            self.assertEqual(export_records([record],root)['scores'][0]['logprob'],values[1,4])
            for update in [dict(sample=''),dict(sample=' '),dict(logprob=None),
                           dict(logprob=False),dict(logprob=float(values[0,4])),
                           dict(scores=dict(path='missing.npy',row=1)),
                           dict(scores=dict(path='rows.npy',row=2)),
                           dict(scores=dict(path='rows.npy',row=True)),
                           dict(scores=dict(path='rows.npy')),dict(token=5),
                           dict(history=[1,5])]:
                with self.subTest(update=update),self.assertRaises(ValueError):
                    export_records([{**record,**update}],root)
            np.save(Path(root)/'unnormalized.npy',values+.1)
            with self.assertRaisesRegex(ValueError,'normalization'):
                export_records([{**record,'scores':dict(path='unnormalized.npy',row=1)}],root)
            with self.assertRaisesRegex(ValueError,'tolerance'):
                export_records([record],root,score_tolerance=float('nan'))

    def test_stable_explicit_derivation_retains_wide_vocab_and_native_values(self):
        logits = np.linspace(-20,20,151936,dtype=np.float32)+100000
        before = logits.copy()
        values = derive_logprobs(logits)
        self.assertEqual(values.dtype,np.float64)
        self.assertAlmostEqual(float(np.exp(values).sum()),1.,places=13)
        np.testing.assert_array_equal(logits,before)
        np.testing.assert_array_equal(values,derive_logprobs(logits-logits.max()))
        matrix = derive_logprobs(np.array([[0.,-np.inf],[1.,2.]],dtype=np.float32),dtype='float32')
        self.assertEqual(matrix.dtype,np.float32)
        self.assertTrue(np.isneginf(matrix[0,1]))
        for bad in [np.array([-np.inf,-np.inf]),np.array([np.inf,0.]),
                    np.array([np.nan,0.]),np.array([],dtype=np.float64),np.array([1,2])]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):derive_logprobs(bad)
        with self.assertRaisesRegex(ValueError,'represented'):
            derive_logprobs(np.array([1e300,1e300]),dtype='float32')
        with self.assertRaisesRegex(ValueError,'represented'):
            derive_logprobs(np.array([-1e308,1e308]))

    def test_native_generation_anchor_rejects_first_context_callback_row(self):
        context = np.arange(12,dtype=np.float32).reshape(3,4)
        generation = np.vstack([context[-1],np.array([[12.,13.,14.,15.],[16.,17.,18.,19.]])])
        original = generation.copy()
        checked = validate_generation_scores(context,generation,generation[1:])
        self.assertTrue(checked['first_generation_is_last_context'])
        self.assertEqual(checked['decode_rows'],2)
        callbacks = generation.copy();callbacks[0] = context[0]
        with self.assertRaisesRegex(ValueError,'last context'):
            validate_generation_scores(context,callbacks)
        with self.assertRaisesRegex(ValueError,'decode'):
            validate_generation_scores(context,generation,generation[1:][::-1])
        with self.assertRaisesRegex(ValueError,'vocabulary'):
            validate_generation_scores(context,generation[:,:3])
        np.testing.assert_array_equal(generation,original)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'native capture requires Torch')
class NativeCaptureTests(unittest.TestCase):
    def test_invalid_metadata_or_integer_logits_do_not_publish_partial_capture(self):
        import torch
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):FixedHistory(' ',[1],[0])
            history = FixedHistory('a',[1],[0])
            logits = torch.tensor([1.,2.])
            for kwargs in [dict(rank=True),dict(rank=-1),dict(boundary=[]),dict(boundary=' ')]:
                with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                    history.capture_and_force(logits,Path(root)/'invalid',**kwargs)
            with self.assertRaises(ValueError):
                history.capture_and_force(torch.tensor([1,2]),Path(root)/'integer')
            self.assertEqual(list(Path(root).iterdir()),[])
            self.assertEqual(history.step,0)
            self.assertEqual(logits.tolist(),[1.,2.])

    def test_explicit_offline_fp64_does_not_round_native_fp64_input(self):
        import torch
        with tempfile.TemporaryDirectory() as root:
            logits = torch.tensor([1000000000000.,1000000000001.,1000000000002.],dtype=torch.float64)
            original = logits.numpy().copy()
            record = FixedHistory('fp64',[1],[2],logprob_dtype='float64').capture_and_force(logits,Path(root)/'fp64')
            expected = derive_logprobs(original)
            np.testing.assert_array_equal(np.load(Path(root)/'fp64.npy'),expected)
            self.assertEqual(record['logits_dtype'],'torch.float64')
            self.assertEqual(record['logprob_dtype'],'float64')
            self.assertIn('Offline',record['normalization'])
            self.assertEqual(record['logprob'],float(expected[2]))
            self.assertEqual(logits.tolist(),[-float('inf'),-float('inf'),0.])

    def test_capture_before_force_completed_callback_and_request_isolation(self):
        import torch
        with tempfile.TemporaryDirectory() as root:
            first = FixedHistory('a', [1, 2], [2])
            second = FixedHistory('b', [1], [1])
            logits = torch.tensor([0., 1., 2.], dtype=torch.bfloat16)
            expected = logits.float().log_softmax(-1).numpy()
            record = first.capture_and_force(logits, Path(root)/'first')
            np.testing.assert_array_equal(np.load(Path(root)/'first.npy'), expected)
            self.assertEqual(json.loads((Path(root)/'first.json').read_text()), record)
            self.assertEqual(record['logits_dtype'], 'torch.bfloat16')
            self.assertEqual(record['history'], [1, 2])
            self.assertEqual(logits.tolist(), [-float('inf'), -float('inf'), 0.])
            untouched = torch.tensor([3., 2., 1.])
            self.assertIsNone(first.capture_and_force(untouched, Path(root)/'extra'))
            self.assertEqual(untouched.tolist(), [3., 2., 1.])
            self.assertEqual(second.step, 0)
            second.capture_and_force(untouched, Path(root)/'second')
            self.assertEqual(second.step, 1)

    def test_zero_probability_and_output_collision(self):
        import torch
        with tempfile.TemporaryDirectory() as root:
            history = FixedHistory('a', [1], [0])
            record = history.capture_and_force(torch.tensor([-float('inf'), 0.]), Path(root)/'zero')
            self.assertIsNone(record['logprob'])
            self.assertTrue(np.isneginf(np.load(Path(root)/'zero.npy')[0]))
            other = FixedHistory('b', [1], [0])
            logits = torch.tensor([1., 0.])
            with self.assertRaises(FileExistsError):
                other.capture_and_force(logits, Path(root)/'zero')
            self.assertEqual(other.step, 0)
            self.assertEqual(logits.tolist(), [1., 0.])


if __name__ == '__main__':
    unittest.main()
