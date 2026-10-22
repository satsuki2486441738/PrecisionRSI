"""Fresh-interpreter initialization, real import identity and interface checks."""
import hashlib
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

    def test_selected_module_fingerprint_tracks_source_with_same_version(self):
        source = self.root / 'probe_fixture.py'
        spec = dict(python_paths=[str(self.root)], imports=['probe_fixture'],
                    sources=[{'module': 'probe_fixture'}])
        before = probe_runtime(spec)
        self.assertEqual(before['status'], 'completed', before)
        self.assertEqual(before['sources'], [dict(module='probe_fixture',
                         path=str(source.resolve()), bytes=source.stat().st_size,
                         sha256=hashlib.sha256(source.read_bytes()).hexdigest())])
        source.write_text(source.read_text() + '\n# isolated candidate source\n')
        after = probe_runtime(spec)
        self.assertEqual(after['status'], 'completed', after)
        self.assertEqual(before['modules'], after['modules'])
        self.assertNotEqual(before['sources'][0]['sha256'], after['sources'][0]['sha256'])
        self.assertEqual(after['sources'][0]['sha256'], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_loaded_module_identity_wins_over_search_path(self):
        import types
        from unittest.mock import patch
        loaded = self.root / 'loaded.py'
        loaded.write_text('# actual loaded implementation\n')
        (self.root / 'native_fixture.py').write_text('raise AssertionError("shadow imported")\n')
        module = types.ModuleType('native_fixture')
        module.__file__ = loaded
        with patch.dict(sys.modules, native_fixture=module):
            report = inspect_runtime(dict(python_paths=[str(self.root)],
                                          sources=[{'module': 'native_fixture'}]))
        self.assertEqual(report['status'], 'completed', report)
        self.assertEqual(report['sources'][0]['path'], str(loaded.resolve()))
        self.assertEqual(report['sources'][0]['sha256'], hashlib.sha256(loaded.read_bytes()).hexdigest())

    def test_explicit_file_hashes_binary_bytes_without_execution(self):
        source = self.root / 'selected.bin'
        content = b'raise AssertionError("must not execute")\n\x00\xff' * 60000
        source.write_bytes(content)
        report = inspect_runtime(dict(sources=[{'path': str(source)}]))
        self.assertEqual(report['status'], 'completed', report)
        self.assertEqual(report['sources'], [dict(path=str(source.resolve()), bytes=len(content),
                         sha256=hashlib.sha256(content).hexdigest())])
        self.assertEqual(report['modules'], {})
        self.assertNotIn('sources', inspect_runtime({}))

    def test_invalid_or_unavailable_sources_fail_with_partial_evidence(self):
        source = self.root / 'retained.py'
        source.write_text('# retained evidence\n')
        marker = self.root / 'unexpected-import'
        (self.root / 'unselected_fixture.py').write_text(
            'from pathlib import Path\nPath(' + repr(str(marker)) + ').touch()\n')
        invalid = [None, 'json', {}, {'module': 'json', 'path': str(source)},
                   {'path': 'relative.py'}, {'path': ''}, {'path': None},
                   {'path': str(self.root)}, {'path': str(self.root / 'missing.py')},
                   {'module': 'sys'}, {'module': 'unselected_fixture'}]
        for selection in invalid:
            with self.subTest(selection=selection):
                report = inspect_runtime(dict(python_paths=[str(self.root)], imports=['json'],
                    sources=[{'path': str(source)}, selection]))
                self.assertEqual(report['status'], 'failed', report)
                self.assertEqual(len(report['sources']), 1)
                self.assertEqual(report['sources'][0]['sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
                self.assertIn('json', report['modules'])
                self.assertIn('Traceback', report['error']['traceback'])
        self.assertFalse(marker.exists())
        for selections in [None, 'json', {'module': 'json'}]:
            with self.subTest(sources=selections):
                self.assertEqual(inspect_runtime({'sources': selections})['status'], 'failed')

    def test_changing_file_does_not_produce_a_successful_fingerprint(self):
        from unittest.mock import patch
        source = self.root / 'changing.py'
        source.write_bytes(b'original file contents')
        sha256 = hashlib.sha256
        class ChangingHash:
            def __init__(self):
                self.inner = sha256()
            def update(self, block):
                self.inner.update(block)
                source.write_bytes(b'replaced with a different file length')
            def hexdigest(self):
                return self.inner.hexdigest()
        with patch('runtime_probe.hashlib.sha256', ChangingHash):
            report = inspect_runtime(dict(sources=[{'path': str(source)}]))
        self.assertEqual(report['status'], 'failed', report)
        self.assertEqual(report['sources'], [])
        self.assertIn('changed', report['error']['message'])

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

    def test_native_report_contains_only_plain_json_values(self):
        import types
        from unittest.mock import patch
        class Version(str):
            pass
        class Count(int):
            pass
        class Number(float):
            pass
        module = types.ModuleType('native_fixture')
        module.__version__ = Version('2.9.0')
        module.__file__ = self.root / 'native.py'
        original = {'version': module.__version__, 'shape': (Count(2), Count(3)),
                    'value': Number(1.25), 'optional': None}
        module.check = lambda: original
        with patch.dict(sys.modules, native_fixture=module):
            report = inspect_runtime(dict(imports=['native_fixture'], checks=['native_fixture:check']))
            self.assertEqual(report['status'], 'completed', report)
            self.assertEqual(report['modules']['native_fixture'],
                             dict(path=str(module.__file__), version='2.9.0'))
            result = report['checks']['native_fixture:check']
            self.assertIs(type(result['version']), str)
            self.assertIs(type(result['shape']), list)
            self.assertTrue(all(type(x) is int for x in result['shape']))
            self.assertIs(type(result['value']), float)
            self.assertEqual(result, dict(version='2.9.0', shape=[2, 3], value=1.25, optional=None))
            self.assertIsInstance(original['shape'], tuple)
            module.__version__ = module.__file__ = None
            report = inspect_runtime(dict(imports=['native_fixture']))
            self.assertEqual(report['modules']['native_fixture'], dict(path=None, version=None))

    def test_native_invalid_check_retains_json_failure_diagnostics(self):
        import types
        from unittest.mock import patch
        module = types.ModuleType('invalid_fixture')
        for value, error in [(object(), 'TypeError'), (float('nan'), 'ValueError')]:
            module.check = lambda: value
            with self.subTest(error=error), patch.dict(sys.modules, invalid_fixture=module):
                report = inspect_runtime(dict(checks=['invalid_fixture:check']))
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['error']['type'], error)
                self.assertIn('Traceback', report['error']['traceback'])
                self.assertEqual(report['checks'], {})
                json.dumps(report, allow_nan=False)

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
        spec.write_text(json.dumps(dict(python_paths=['.'], imports=['probe_fixture'],
                                       sources=[{'path': 'probe_fixture.py'}, {'module': 'probe_fixture'}])))
        output = self.root/'report.json'
        command = [sys.executable, '-B', str(TOOLS/'runtime_probe.py'), '--spec', str(spec), '--output', str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report['sources'][0]['path'], str((self.root/'probe_fixture.py').resolve()))
        self.assertEqual(report['sources'][0]['sha256'], report['sources'][1]['sha256'])
        before = output.read_bytes()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
