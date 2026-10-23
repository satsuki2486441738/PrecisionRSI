"""One-connection artifact retrieval with size limits and exact verified bytes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'skills/precision-compare/scripts'
sys.path.insert(0, str(TOOLS))
from evidence_bundle import fetch


class EvidenceFetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'capture'
        self.source.mkdir()
        self.output = self.root / 'download'
        self.transport = [sys.executable, '-I', '-B', '-']

    def collect(self, include=None, **kwargs):
        return fetch(self.transport, str(self.source), include or ['*.json'], self.output, **kwargs)

    def test_batch_exact_bytes_index_and_large_arrays_stay_at_source(self):
        files = {'scores.json': b'{"target":-1.2345}\n',
                 'rank0/status.json': b'{"status":"completed"}',
                 'rank0/stderr.bin': b'log\x00\xff'}
        for name, data in files.items():
            p = self.source/name
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(data)
        with (self.source/'large.pt').open('wb') as stream:
            stream.seek(100 * 1024**2)
            stream.write(b'x')
        original = subprocess.Popen
        with patch('evidence_bundle.subprocess.Popen', wraps=original) as launch:
            result = self.collect(['*.json', 'rank0/*.json', 'rank0/*.bin'], max_file_bytes=1024)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(result['action'], 'fetch')
        self.assertEqual(result['file_count'], 3)
        self.assertNotIn('bundle', result)
        self.assertLess(result['transfer_bytes'], 4096)
        self.assertEqual(result['payload_bytes'], sum(map(len, files.values())))
        index = json.loads((self.output/'EXPORT.json').read_text())
        self.assertEqual({x['path'] for x in index['files']}, set(files))
        for entry in index['files']:
            data = files[entry['path']]
            self.assertEqual((self.output/entry['path']).read_bytes(), data)
            self.assertEqual(entry['exported_sha256'], hashlib.sha256(data).hexdigest())
        self.assertFalse((self.output/'large.pt').exists())
        self.assertFalse(list(self.output.glob('*.zip')))

    def test_each_cap_and_empty_selection_fail_without_partial_output(self):
        (self.source/'data.json').write_bytes(bytes(range(256))*16)
        for options, message in [({'max_file_bytes':100},'max_file_bytes'),
                                 ({'max_expanded_bytes':100},'cap'),
                                 ({'max_transfer_bytes':100},'transfer-byte'),
                                 ({'include':['missing*']},'selected no files')]:
            with self.subTest(options=options):
                include=options.get('include',['*.json'])
                kwargs={k:v for k,v in options.items() if k!='include'}
                with self.assertRaises(RuntimeError) as error:
                    self.collect(include, **kwargs)
                self.assertIn(message,str(error.exception))
                self.assertFalse(self.output.exists())

    def test_noisy_transport_stops_at_receive_cap_without_waiting_for_timeout(self):
        code=('import os,sys,time;sys.stdin.buffer.read();'
              'os.write(2,b"e"*(4*1024**2));os.write(1,b"x"*(4*1024**2));time.sleep(30)')
        started=time.monotonic()
        with self.assertRaisesRegex(ValueError,'transfer-byte cap'):
            fetch([sys.executable,'-c',code],str(self.source),['*.json'],self.output,
                  timeout=2,max_transfer_bytes=1024)
        self.assertLess(time.monotonic()-started,2)
        self.assertFalse(self.output.exists())

    def test_existing_output_and_invalid_budgets_do_not_launch(self):
        (self.source/'data.json').write_text('{}')
        for key, value in [('timeout',0),('timeout',True),('timeout',float('nan')),
                           ('max_file_bytes',0),('max_transfer_bytes',False)]:
            with self.subTest(key=key), patch('evidence_bundle.subprocess.Popen') as launch:
                with self.assertRaises(ValueError):self.collect(**{key:value})
                launch.assert_not_called()
        self.output.mkdir()
        with patch('evidence_bundle.subprocess.Popen') as launch:
            with self.assertRaises(FileExistsError):self.collect()
            launch.assert_not_called()

    @unittest.skipUnless(os.name=='posix','symlinks')
    def test_selected_links_and_traversal_rejected(self):
        outside=self.root/'outside';outside.mkdir();(outside/'private.json').write_text('{}')
        (self.source/'link').symlink_to(outside,target_is_directory=True)
        for pattern in ['link/*.json','../outside/*.json']:
            with self.subTest(pattern=pattern),self.assertRaises(RuntimeError):self.collect([pattern])
            self.assertFalse(self.output.exists())

    def test_timeout_terminates_transport_and_does_not_create_output(self):
        started=time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            fetch([sys.executable,'-c','import time;time.sleep(30)'],
                  str(self.source),['*.json'],self.output,timeout=.2)
        self.assertLess(time.monotonic()-started,3)
        self.assertFalse(self.output.exists())

    def test_cli_request_and_corrupt_transport(self):
        (self.source/'scores.json').write_text('{"p":-2}')
        request=self.root/'request.json'
        request.write_text(json.dumps(dict(transport=self.transport,source_dir=str(self.source),
                                          include=['*.json'])))
        done=subprocess.run([sys.executable,'-I','-B',str(TOOLS/'evidence_bundle.py'),
            'fetch',str(request),str(self.output)],capture_output=True,text=True)
        self.assertEqual(done.returncode,0,done.stderr)
        self.assertEqual(json.loads(done.stdout)['file_count'],1)
        request.write_text(json.dumps(dict(transport=[sys.executable,'-c','print("bad archive")'],
            source_dir=str(self.source),include=['*.json'])))
        bad=self.root/'bad'
        done=subprocess.run([sys.executable,'-I','-B',str(TOOLS/'evidence_bundle.py'),
            'fetch',str(request),str(bad)],capture_output=True,text=True)
        self.assertEqual(done.returncode,1)
        self.assertFalse(bad.exists())


if __name__=='__main__':unittest.main()
