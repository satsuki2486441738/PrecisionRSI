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

TOOLS = Path(__file__).resolve().parents[1] / 'skills/precision-compare/scripts'
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

    def test_execution_reuses_activation_prefix_and_literal_environment(self):
        envbin = self.root / 'environment' / 'bin'
        envbin.mkdir(parents=True)
        tool = envbin / 'native-tool'
        tool.write_text('#!' + sys.executable + '\nimport os;print(os.environ["SETTING"])\n')
        tool.chmod(0o755)
        profile = self.root / 'execution.json'
        profile.write_text(json.dumps(dict(argv_prefix=[sys.executable, '-c',
            'import os,subprocess,sys;assert "REMOVED" not in os.environ;'
            'raise SystemExit(subprocess.call(sys.argv[1:]))'],
            env={'SETTING': '$(literal) $HOME', 'REMOVED': None},path_prepend=[str(envbin)])))
        with patch.dict(os.environ, {'REMOVED': 'inherited'}):
            for task in ('preflight', 'capture'):
                folder, result = record_run(self.root/'runs', 'cpu', task, 5,
                    ['native-tool'], execution=profile)
                self.assertEqual(result['status'], 'completed', (folder/'stderr.bin').read_text())
                self.assertEqual((folder/'stdout.bin').read_text(), '$(literal) $HOME\n')
                self.assertEqual(result['execution']['path'], str(profile.resolve()))
                self.assertEqual(result['argv'][-1], 'native-tool')
                self.assertNotIn('env', result)

    def test_execution_snapshot_survives_profile_changes(self):
        import hashlib
        profile=self.root/'execution.json'
        profile.write_text(json.dumps(dict(env={'NATIVE_READY':'original'})))
        folder,result=record_run(self.root/'runs','cpu','snapshot',2,
            [sys.executable,'-c','pass'],execution=profile)
        profile.write_text(json.dumps(dict(env={'NATIVE_READY':'changed'})))
        snapshot=Path(result['execution']['snapshot'])
        self.assertEqual(snapshot.parent,folder)
        self.assertEqual(json.loads(snapshot.read_text())['env']['NATIVE_READY'],'original')
        self.assertEqual(result['execution']['sha256'],hashlib.sha256(snapshot.read_bytes()).hexdigest())

    def test_execution_cli_and_retained_status_need_no_relaunch(self):
        from record_run import summarize_run
        profile=self.root/'execution.json'
        profile.write_text(json.dumps(dict(argv_prefix=[sys.executable,'-c'],env={'NATIVE':'ready'})))
        result=subprocess.run([sys.executable,'-B',str(TOOLS/'record_run.py'),
            '--execution',str(profile),'--output-root',str(self.root/'cli'),
            '--side','cpu','--task','profile','--timeout','2','--',
            'import os;print(os.environ["NATIVE"]);raise SystemExit(3)'],capture_output=True,text=True)
        self.assertEqual(result.returncode,3,result.stderr)
        reported=json.loads(result.stdout)
        with patch('record_run.subprocess.Popen',side_effect=AssertionError('unexpected launch')):
            retained=summarize_run(reported['run'])
        self.assertEqual(reported,retained)
        self.assertEqual(retained['logs']['stdout']['tail'],'ready\n')

    def test_invalid_execution_fails_before_launch_or_run_creation(self):
        profile = self.root/'execution.json'
        for spec in ({'environ': {}}, {'env': {'BAD=KEY': 'x'}},
                     {'env': {'X': 2}}, {'path_prepend': ['relative']},
                     {'argv_prefix': ['']}, {'argv_prefix': 'shell'}):
            profile.write_text(json.dumps(spec))
            with self.subTest(spec=spec), patch('record_run.subprocess.Popen') as launch:
                with self.assertRaises(ValueError):
                    record_run(self.root/'runs', 'cpu', 'invalid', 2,
                               [sys.executable, '-c', 'pass'], execution=profile)
                launch.assert_not_called()
                self.assertFalse((self.root/'runs').exists())

    def test_argv_factory_uses_reserved_run_directory(self):
        folder, result = record_run(self.root / 'runs', 'cpu', 'factory', 5,
            lambda run: [sys.executable, '-c', 'import sys;open(sys.argv[1],"w").write("native")',
                         str(run / 'artifact.txt')], cwd=self.root)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual((folder / 'artifact.txt').read_text(), 'native')
        self.assertEqual(result['argv'][-1], str(folder / 'artifact.txt'))

    def test_invalid_factory_retains_failure_without_starting_child(self):
        with patch('record_run.subprocess.Popen') as launch:
            folder, result = record_run(self.root / 'runs', 'cpu', 'factory', 5,
                                        lambda run: [123], cwd=self.root)
        launch.assert_not_called()
        self.assertEqual(result['status'], 'launch_failed')
        self.assertEqual(result['returncode'], 127)
        self.assertEqual(json.loads((folder / 'command.json').read_text()), result)

    def test_preparation_exception_is_retained_without_launch(self):
        def prepare(run):
            raise RuntimeError('cannot prepare arguments')
        with patch('record_run.subprocess.Popen') as launch:
            folder, result = record_run(self.root / 'runs', 'cpu', 'factory', 5, prepare)
        launch.assert_not_called()
        self.assertEqual(result['status'], 'launch_failed')
        self.assertEqual(result['error']['type'], 'RuntimeError')
        self.assertTrue((folder / 'command.json').exists())

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

    def cli_child(self, task, code, *options):
        return subprocess.run(
            [sys.executable, '-B', str(TOOLS/'record_run.py'),
             '--output-root', str(self.root/'cli'), '--side', 'cpu', '--task', task,
             '--timeout', '2', *options, '--', sys.executable, '-c', code],
            capture_output=True, text=True)

    def test_cli_failure_has_bounded_binary_tails_and_complete_logs(self):
        result = self.cli_child('binary',
            'import os; os.write(1,b"x"*100000+b"END"); '
            'os.write(2,b"prefix"+bytes([255])+b"error"); raise SystemExit(7)',
            '--tail-bytes', '8')
        self.assertEqual(result.returncode, 7, result.stderr)
        report = json.loads(result.stdout)
        for name, expected in [('stdout', b'x'*100000+b'END'),
                               ('stderr', b'prefix'+bytes([255])+b'error')]:
            entry = report['logs'][name]
            self.assertEqual(entry['bytes'], len(expected))
            self.assertEqual(Path(entry['path']).read_bytes(), expected)
            self.assertEqual(entry['tail'], expected[-8:].decode('utf-8', errors='replace'))
            self.assertTrue(entry['truncated'])
        self.assertLess(len(result.stdout), 2000)
        self.assertNotIn('logs', json.loads((Path(report['run'])/'command.json').read_text()))

    def test_cli_success_and_disabled_tails_keep_output_in_files(self):
        for task, exitcode, options in [('success', 0, []), ('disabled', 7, ['--tail-bytes', '0'])]:
            with self.subTest(task=task):
                result = self.cli_child(task,
                    f'import os; os.write(1,b"x"*100000); raise SystemExit({exitcode})', *options)
                self.assertEqual(result.returncode, exitcode)
                report = json.loads(result.stdout)
                self.assertEqual(report['logs']['stdout']['bytes'], 100000)
                self.assertNotIn('tail', report['logs']['stdout'])
                self.assertLess(len(result.stdout), 2000)

    def test_cli_timeout_and_launch_failure_diagnostics(self):
        result = self.cli_child('timeout',
            'import sys,time; print("waiting", file=sys.stderr, flush=True); time.sleep(30)',
            '--timeout', '0.5')
        self.assertEqual(result.returncode, 124, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['logs']['stderr']['tail'], 'waiting\n')
        self.assertFalse(report['logs']['stderr']['truncated'])
        result = subprocess.run(
            [sys.executable, str(TOOLS/'record_run.py'), '--output-root', str(self.root/'cli'),
             '--side', 'cpu', '--task', 'absent', '--timeout', '2', '--', str(self.root/'missing')],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 127)
        report = json.loads(result.stdout)
        self.assertEqual(report['error']['type'], 'FileNotFoundError')
        self.assertEqual(report['logs']['stderr']['bytes'], 0)
        self.assertEqual(report['logs']['stderr']['tail'], '')

    def test_cli_invalid_tail_limit_never_launches(self):
        result = self.cli_child('invalid', 'open("unexpected", "w").close()', '--tail-bytes', '-1')
        self.assertEqual(result.returncode, 2)
        self.assertIn('nonnegative', result.stderr)
        self.assertFalse((self.root/'cli').exists())

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
