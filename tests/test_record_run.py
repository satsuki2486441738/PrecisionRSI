"""Behavioral regressions for literal execution and durable bounded run capture."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'skills/precision-skill/scripts'
sys.path.insert(0, str(TOOLS))
from record_run import record_run


class RecordRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_child(self, code, *args, timeout=5):
        return record_run(self.root / 'runs', 'cpu', 'test', timeout,
                          [sys.executable, '-c', code, *args], cwd=self.root)

    def test_success_literal_argv_binary_streams_and_metadata(self):
        literal = '$(touch injected); `touch injected` $HOME * ; &'
        folder, result = self.run_child(
            'import os,sys; print(sys.argv[1]); print(os.getcwd()); '
            'sys.stdout.flush(); os.write(2,b"err\\x00\\xff")', literal)
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual((folder / 'stdout.bin').read_text().splitlines(), [literal, str(self.root.resolve())])
        self.assertEqual((folder / 'stderr.bin').read_bytes(), b'err\x00\xff')
        self.assertFalse((self.root / 'injected').exists())
        self.assertEqual(json.loads((folder / 'command.json').read_text()), result)
        self.assertIsInstance(result['child_pid'], int)
        self.assertGreaterEqual(result['wall_seconds'], 0)
        self.assertGreaterEqual(result['end'], result['start'])
        self.assertNotIn('environment', result)
        self.assertRegex(folder.name, r'^cpu-test-\d{8}T\d{6}Z$')

    def test_nonzero_retains_output_and_exit_code(self):
        folder, result = self.run_child('import sys; print("failure"); sys.exit(7)')
        self.assertEqual((result['status'], result['returncode'], result['child_returncode']), ('failed', 7, 7))
        self.assertEqual((folder / 'stdout.bin').read_bytes(), b'failure\n')

    def test_missing_executable_records_launch_failure(self):
        folder, result = record_run(self.root / 'runs', 'a', 'missing', 5,
                                    [str(self.root / 'absent-executable')])
        self.assertEqual((result['status'], result['returncode']), ('launch_failed', 127))
        self.assertIsNone(result['child_pid'])
        self.assertEqual(result['error']['type'], 'FileNotFoundError')
        self.assertEqual((folder / 'stderr.bin').read_bytes(), b'')
        self.assertEqual(json.loads((folder / 'command.json').read_text()), result)

    @unittest.skipUnless(os.name == 'posix', 'POSIX signals')
    def test_signal_exit_is_normalized(self):
        _, result = self.run_child('import os,signal; os.kill(os.getpid(),signal.SIGTERM)')
        self.assertEqual(result['child_returncode'], -signal.SIGTERM)
        self.assertEqual(result['returncode'], 128 + signal.SIGTERM)

    @unittest.skipUnless(os.name == 'posix', 'POSIX process group cleanup')
    def test_timeout_kills_descendant_even_when_leader_exits_on_term(self):
        # Descendant writes a heartbeat and ignores SIGTERM. Cleanup must kill it
        # even though the direct child exits promptly on SIGTERM.
        heartbeat = self.root / 'heartbeat'
        descendant = ('import signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                      f'p=pathlib.Path({str(heartbeat)!r}); '
                      '\nwhile True:\n with p.open("ab") as f: f.write(b"x")\n time.sleep(.02)\n')
        code = ('import subprocess,sys,time,pathlib; '
                f'p=subprocess.Popen([sys.executable,"-c",{descendant!r}]); '
                'print(p.pid,flush=True); time.sleep(60)')
        folder, result = self.run_child(code, timeout=0.8)
        self.assertEqual((result['status'], result['returncode']), ('timed_out', 124))
        self.assertTrue(heartbeat.exists(), 'descendant must actually run')
        size = heartbeat.stat().st_size
        time.sleep(0.15)
        self.assertEqual(heartbeat.stat().st_size, size, 'descendant survived recorder timeout')
        self.assertTrue((folder / 'stdout.bin').read_text().strip().isdigit())
        self.assertLess(result['wall_seconds'], 3)

    @unittest.skipUnless(os.name == 'posix', 'POSIX forced termination')
    def test_timeout_escalates_when_child_ignores_term(self):
        folder, result = self.run_child(
            'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
            'print("ready",flush=True); time.sleep(60)', timeout=0.4)
        self.assertEqual((result['status'], result['returncode']), ('timed_out', 124))
        self.assertEqual(result['child_returncode'], -signal.SIGKILL)
        self.assertEqual((folder/'stdout.bin').read_bytes(), b'ready\n')
        self.assertLess(result['wall_seconds'], 3)

    def test_invalid_arguments_rejected_before_creation_or_launch(self):
        good = dict(output_root=self.root/'runs', side='cpu', task='check', timeout=1,
                    argv=[sys.executable, '-c', 'raise AssertionError("launched")'])
        cases = ([{field: value} for field in ['side','task']
                  for value in ['../escape','/abs','bad_name','a--b','a/child','','a b']]
                 + [{'timeout': value} for value in [0,-1,float('inf'),float('nan'),True,'invalid']]
                 + [{'argv': value} for value in [[], '', [''], ['a\0b'], [123]]]
                 + [{'cwd': self.root/'absent'}])
        for override in cases:
            with self.subTest(override=override), patch('record_run.subprocess.Popen') as launch:
                with self.assertRaises(ValueError):
                    record_run(**(good | override))
                launch.assert_not_called()
                self.assertFalse((self.root/'runs').exists())

    def test_preexisting_directory_is_preserved_and_never_launched(self):
        fixed = datetime(2026, 10, 22, 1, 2, 3, tzinfo=timezone.utc)
        with patch('record_run._utc_now', return_value=fixed):
            folder, _ = self.run_child('print("historical")')
            original = {p.name:p.read_bytes() for p in folder.iterdir()}
            with patch('record_run.subprocess.Popen') as launch:
                with self.assertRaises(FileExistsError):
                    self.run_child('print("overwrite")')
                launch.assert_not_called()
            self.assertEqual({p.name:p.read_bytes() for p in folder.iterdir()}, original)

    def test_cli_exit_codes_and_invalid_timeout(self):
        command = [sys.executable, '-B', str(TOOLS/'record_run.py'), '--output-root', str(self.root/'cli'),
                   '--side','a','--task','cli','--timeout','2','--']
        result = subprocess.run(command+[sys.executable,'-c','raise SystemExit(9)'], capture_output=True)
        self.assertEqual(result.returncode, 9, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record['returncode'], 9)
        invalid = command.copy()
        invalid[invalid.index('--timeout')+1] = 'nan'
        result = subprocess.run(invalid+[sys.executable,'-c','pass'], capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b'finite and positive', result.stderr)


if __name__ == '__main__':
    unittest.main()
