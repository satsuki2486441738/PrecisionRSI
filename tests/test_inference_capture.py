"""Logical history checks, artifact coordinates and native capture boundaries."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

TOOLS = Path(__file__).resolve().parents[1]/'skills/precision-skill/scripts'
sys.path.insert(0, str(TOOLS))
from inference_capture import FixedHistory, export_records, validate_history


class InferenceExportTests(unittest.TestCase):
    def test_positions_target_shift_zero_probability_and_duplicates(self):
        record = dict(sample='request', position=2, history=[1, 2, 3], token=4,
                      logprob=-1.5, scores=dict(path='row.npy', row=0))
        with tempfile.TemporaryDirectory() as root:
            result = export_records([record], root)
            self.assertEqual(result['positions'][0]['position'], 2)
            self.assertEqual(result['scores'][0]['position'], 3)
            self.assertEqual(result['positions'][0]['scores']['path'], str((Path(root)/'row.npy').resolve()))
            zero = export_records([{**record, 'logprob': None}], root)
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


@unittest.skipUnless(importlib.util.find_spec('torch'), 'native capture requires Torch')
class NativeCaptureTests(unittest.TestCase):
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
