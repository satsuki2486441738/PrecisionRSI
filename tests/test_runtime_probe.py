"""Fresh-interpreter initialization, real import identity and interface checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'skills/precision-skill/scripts'
sys.path.insert(0, str(TOOLS))
from runtime_probe import inspect_runtime, probe_runtime


class RuntimeProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'probe_fixture.py').write_text(
            "import builtins, os, time\n"
            "__version__='actual-module'\n"
            "def initialize(): builtins.precision_ready=True; return {'pid':os.getpid()}\n"
            "def inspect_worker():\n"
            "    assert builtins.precision_ready\n"
            "    print('worker diagnostic')\n"
            "    return {'ready':True,'pid':os.getpid()}\n"
            "def wait(): time.sleep(60)\n")
        info = self.root/'probe_fixture-9.0.dist-info'
        info.mkdir()
        (info/'METADATA').write_text('Metadata-Version: 2.1\nName: probe-fixture\nVersion: 9.0\n')

    def test_fresh_process_and_actual_module_separate_from_distribution(self):
        spec = dict(python_paths=[str(self.root)], imports=['probe_fixture'],
                    symbols=['probe_fixture:initialize'],
                    checks=['probe_fixture:initialize','probe_fixture:inspect_worker'],
                    distributions=['probe-fixture'])
        result = probe_runtime(spec)
        self.assertEqual(result['status'], 'completed', result)
        self.assertNotEqual(result['pid'], __import__('os').getpid())
        self.assertEqual(result['modules']['probe_fixture']['version'], 'actual-module')
        self.assertEqual(result['modules']['probe_fixture']['path'], str(self.root/'probe_fixture.py'))
        self.assertEqual(result['distributions']['probe-fixture']['version'], '9.0')
        self.assertTrue(result['checks']['probe_fixture:inspect_worker']['ready'])
        self.assertIn('worker diagnostic', result['diagnostics'])
        # Initialization in the first worker is not inherited by the next one.
        result = probe_runtime(dict(python_paths=[str(self.root)], checks=['probe_fixture:inspect_worker']))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error']['type'], 'AttributeError')

    def test_worker_inspection_restores_import_paths_on_success_and_failure(self):
        before = sys.path.copy()
        original = sys.path
        try:
            for symbols in ([], ['probe_fixture:removed_helper']):
                result = inspect_runtime(dict(python_paths=[str(self.root)], symbols=symbols))
                self.assertEqual(result['status'], 'failed' if symbols else 'completed')
                self.assertIs(sys.path, original)
                self.assertEqual(sys.path, before)
        finally:
            sys.path[:] = before

    def test_invalid_payload_does_not_launch_worker(self):
        from unittest.mock import patch
        for value, error in [(object(), TypeError), (float('nan'), ValueError)]:
            with self.subTest(value=type(value).__name__):
                with patch('runtime_probe.subprocess.Popen') as launch:
                    with self.assertRaises(error):
                        probe_runtime({'checks': [value]})
                    launch.assert_not_called()

    def test_multibyte_diagnostics_tail(self):
        from unittest.mock import patch
        (self.root/'noisy.py').write_text("print('中文' * 5000)\n", encoding='utf-8')
        # A caller's locale/stdio policy must not corrupt the UTF-8 diagnostic
        # protocol of the fresh worker, including on Windows with a CP936 locale.
        with patch.dict('os.environ', {'PYTHONIOENCODING': 'ascii'}):
            result = probe_runtime(dict(python_paths=[str(self.root)], imports=['noisy']))
        self.assertEqual(result['status'], 'completed', result)
        self.assertGreater(result['diagnostic_bytes'], 8192)
        self.assertTrue(result['diagnostics'].rstrip('\r\n').endswith('中文'))

    def test_missing_deferred_interface_is_reported(self):
        result = probe_runtime(dict(python_paths=[str(self.root)], imports=['probe_fixture'],
                                    symbols=['probe_fixture:removed_helper']))
        self.assertEqual(result['status'], 'failed')
        self.assertIn('removed_helper', result['error']['message'])
        self.assertEqual(result['symbols'], [])

    def test_worker_timeout_is_bounded(self):
        result = probe_runtime(dict(python_paths=[str(self.root)], checks=['probe_fixture:wait']), timeout=.3)
        self.assertEqual(result['status'], 'timeout')
        self.assertIsNotNone(result['returncode'])

    def test_cli_relative_import_path_and_existing_output(self):
        spec = self.root/'spec.json'
        spec.write_text(json.dumps(dict(python_paths=['.'], imports=['probe_fixture'])))
        output = self.root/'report.json'
        command = [sys.executable, '-B', str(TOOLS/'runtime_probe.py'), '--spec', str(spec), '--output', str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        before = output.read_bytes()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
