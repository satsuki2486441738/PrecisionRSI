"""Indexed evidence export, literal transformations and hostile ZIP regressions."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

TOOLS = Path(__file__).resolve().parents[1] / 'skills/precision-skill/scripts'
sys.path.insert(0, str(TOOLS))
import evidence_bundle as bundle


def sha(data):
    return hashlib.sha256(data).hexdigest()


class EvidenceBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.archive = self.root / 'evidence.zip'

    def long_path_fixture(self):
        # Independent native path conversion also makes test cleanup work when
        # Windows' legacy MAX_PATH policy is enabled for the Python process.
        def native(path):
            value = str(path.absolute())
            if os.name == 'nt':
                value = '\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value
            return Path(value)

        self.addCleanup(shutil.rmtree, native(self.root))
        base = self.root
        for number in range(4):
            base /= str(number) + '-' + 'long-path-segment-' * 4
        self.assertGreater(len(str(base)), 300)
        source = base / 'source'
        native(source).mkdir(parents=True)
        member = 'callback-events/event-' + '0123456789abcdef' * 5 + '.log'
        target = native(source.joinpath(*member.split('/')))
        target.parent.mkdir(parents=True)
        data = b'captured event\r\n' + bytes(range(256))
        target.write_bytes(data)
        return native, base, source, member, data

    def write_archive(self, files, index='EXPORT.json', records=None, extras=()):
        if records is None:
            if index == 'EXPORT.json':
                records = dict(files=[dict(path=name, exported_sha256=sha(data),
                                          exported_bytes=len(data)) for name, data in files.items()])
            else:
                records = {name: dict(sha256=sha(data), bytes=len(data)) for name, data in files.items()}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(self.archive, 'w', zipfile.ZIP_DEFLATED) as archive:
                for name, data in list(files.items()) + list(extras):
                    archive.writestr(name, data)
                archive.writestr(index, json.dumps(records).encode())

    def test_binary_exactness_and_streamed_large_file(self):
        data = bytes(range(256)) * 32769
        (self.source / 'weights.npy').write_bytes(data)
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('full-file read')):
            receipt = bundle.pack(self.source, self.archive, {'secret': '[HOST]'})
            verified = bundle.verify(self.archive)
            extracted = bundle.extract(self.archive, self.root / 'out')
        self.assertEqual(receipt['archive_sha256'], verified['archive_sha256'])
        self.assertEqual(extracted['payload_bytes'], len(data))
        self.assertEqual((self.root / 'out/weights.npy').read_bytes(), data)
        with zipfile.ZipFile(self.archive) as archive:
            entry = json.loads(archive.read('EXPORT.json'))['files'][0]
        self.assertEqual(entry['original_sha256'], sha(data))
        self.assertEqual(entry['exported_sha256'], sha(data))
        self.assertEqual(entry['literal_replacement_count'], 0)
        self.assertEqual(verified['size_source'], 'index')

    def test_explicit_utf8_literals_all_extensions_and_no_patterns_in_index(self):
        secret = 'private-host-\u79d8\u5bc6'
        data = (secret + '\r\npackage: 10.14.1.48\n').encode()
        for name in ('config.yaml', 'run.sh', 'worker.log', 'extension.custom'):
            (self.source / name).write_bytes(data)
        bundle.pack(self.source, self.archive, [(secret, '[HOST]')])
        with zipfile.ZipFile(self.archive) as archive:
            index_data = archive.read('EXPORT.json')
            index = json.loads(index_data)
            self.assertNotIn(secret.encode(), index_data)
            self.assertNotIn(b'[HOST]', index_data)
            self.assertEqual(len(index['files']), 4)
            for entry in index['files']:
                exported = archive.read(entry['path'])
                self.assertEqual(exported, data.replace(secret.encode(), b'[HOST]'))
                self.assertIn(b'10.14.1.48', exported)
                self.assertEqual(entry['original_sha256'], sha(data))
                self.assertEqual(entry['original_bytes'], len(data))
                self.assertEqual(entry['exported_sha256'], sha(exported))
                self.assertEqual(entry['exported_bytes'], len(exported))
                self.assertEqual(entry['literal_replacement_count'], 1)

    def test_replacement_boundaries_and_sequential_semantics(self):
        text = ('ab\u79d8\u5bc6c-secret-' * 11) + 'secret'
        pairs = [('secret', 'abc'), ('abc', 'X'), ('\u79d8\u5bc6', 'Y')]
        data = text.encode()
        (self.source / 'run.log').write_bytes(data)
        expected = text
        for pattern, replacement in pairs:
            expected = expected.replace(pattern, replacement)
        with patch.object(bundle, 'CHUNK_BYTES', 5):
            bundle.pack(self.source, self.archive, pairs)
            bundle.verify(self.archive)
        with zipfile.ZipFile(self.archive) as archive:
            self.assertEqual(archive.read('run.log'), expected.encode())

    def test_protected_extensions_nul_and_invalid_utf8_never_transformed(self):
        values = {name: b'secret' for name in ('a.npy', 'a.npz', 'a.safetensors', 'a.pt',
                                             'a.pth', 'a.bin', 'a.gguf', 'a.f16',
                                             'a.bf16', 'a.f32', 'a.f64')}
        values.update({'with-nul.log': b'secret\0payload', 'invalid.yaml': b'secret\xffpayload'})
        for name, data in values.items():
            (self.source / name).write_bytes(data)
        with patch.object(bundle, 'CHUNK_BYTES', 3):
            bundle.pack(self.source, self.archive, {'secret': '[HOST]'})
        with zipfile.ZipFile(self.archive) as archive:
            for name, data in values.items():
                self.assertEqual(archive.read(name), data)

    def test_tamper_and_declared_size_mismatch(self):
        original = b'original'
        records = dict(files=[dict(path='data', exported_sha256=sha(original), exported_bytes=len(original))])
        self.write_archive({'data': b'modified'}, records=records)
        with self.assertRaisesRegex(ValueError, 'hash or byte count'):
            bundle.verify(self.archive)
        with self.assertRaises(ValueError):
            bundle.extract(self.archive, self.root / 'bad')
        self.assertFalse((self.root / 'bad').exists())
        records['files'][0]['exported_bytes'] += 1
        self.write_archive({'data': original}, records=records)
        with self.assertRaisesRegex(ValueError, 'byte count mismatch'):
            bundle.verify(self.archive)

    def test_path_traversal_absolute_windows_and_nonportable_names(self):
        names = ['../escape', '/absolute', 'C:/drive', 'a\\b', 'a//b', './x',
                 'nul.txt', 'aux', 'COM1.log', 'con .txt', 'x.', 'x ', 'a:b', 'a/../b', 'a?b']
        for name in names:
            with self.subTest(name=name):
                self.write_archive({name: b'x'})
                with self.assertRaises(ValueError):
                    bundle.extract(self.archive, self.root / 'out')
                self.assertFalse((self.root / 'out').exists())

    def test_duplicates_case_unicode_and_file_directory_collisions(self):
        for files, extras in [({'x': b'a'}, [('x', b'a')]),
                              ({'x': b'a', 'X': b'a'}, []),
                              ({'x': b'a', 'x/child': b'a'}, []),
                              ({'\u00e9': b'a', 'e\u0301': b'a'}, [])]:
            with self.subTest(names=list(files)):
                self.write_archive(files, extras=extras)
                with self.assertRaisesRegex(ValueError, 'colli|duplicate'):
                    bundle.verify(self.archive)

    def test_symlink_and_directory_members_rejected(self):
        for mode, name, attributes in [(stat.S_IFLNK | 0o777, 'link', 0),
                                        (stat.S_IFDIR | 0o755, 'dir/', 0x10),
                                        (stat.S_IFIFO | 0o600, 'pipe', 0)]:
            with self.subTest(mode=mode):
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = mode << 16 | attributes
                self.write_archive({}, extras=[(info, b'target')])
                with self.assertRaises(ValueError):
                    bundle.verify(self.archive)

    def test_missing_unindexed_self_index_and_duplicate_records_rejected(self):
        cases = [dict(files=[]),
                 dict(files=[dict(path='missing', exported_sha256=sha(b'x'), exported_bytes=1)]),
                 dict(files=[dict(path='EXPORT.json', exported_sha256=sha(b'x'), exported_bytes=1)]),
                 dict(files=[dict(path='data', exported_sha256=sha(b'x'))] * 2)]
        for records in cases:
            with self.subTest(records=records):
                self.write_archive({'data': b'x'}, records=records)
                with self.assertRaises(ValueError):
                    bundle.verify(self.archive)
        self.write_archive({}, index='OTHER.json')
        with self.assertRaisesRegex(ValueError, 'missing archive index'):
            bundle.verify(self.archive)

    def test_files_index_and_legacy_export_size_disclosure(self):
        self.write_archive({'x/data': b'numerical'}, index='FILES.json')
        receipt = bundle.extract(self.archive, self.root / 'out', index='FILES.json')
        self.assertEqual(receipt['size_source'], 'index')
        self.assertEqual((self.root / 'out/x/data').read_bytes(), b'numerical')
        self.write_archive({'data': b'x'}, records=dict(files=[dict(path='data', exported_sha256=sha(b'x'))]))
        receipt = bundle.verify(self.archive)
        self.assertEqual(receipt['entries_without_indexed_sizes'], 1)
        self.assertEqual(receipt['size_source'], 'zip_metadata')
        self.write_archive({'data': b'x'}, index='FILES.json', records={'data': dict(sha256=sha(b'x'))})
        with self.assertRaisesRegex(ValueError, 'byte count'):
            bundle.verify(self.archive, index='FILES.json')

    def test_invalid_index_hash_size_and_duplicate_json_keys(self):
        for digest, size in [('not-a-hash', 1), (sha(b'x'), True), (sha(b'x'), -1)]:
            self.write_archive({'data': b'x'}, records=dict(files=[dict(path='data', exported_sha256=digest, exported_bytes=size)]))
            with self.assertRaises(ValueError):
                bundle.verify(self.archive)
        with zipfile.ZipFile(self.archive, 'w') as archive:
            archive.writestr('EXPORT.json', '{"files":[],"files":[]}')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
            bundle.verify(self.archive)

    def test_nonfinite_numbers_rejected_even_in_ignored_nested_metadata(self):
        for number in ('1e999', '-1e999', 'Infinity', '-Infinity', 'NaN'):
            with self.subTest(number=number):
                with zipfile.ZipFile(self.archive, 'w') as archive:
                    archive.writestr('EXPORT.json',
                                     '{"files":[],"metadata":{"ignored":[' + number + ']}}')
                with self.assertRaisesRegex(ValueError, 'nonfinite JSON'):
                    bundle.verify(self.archive)

    def test_cap_applies_before_payload_reads_and_extraction(self):
        self.write_archive({'large': b'x' * 100000})
        with patch.object(zipfile.ZipFile, 'open', side_effect=AssertionError('payload opened')):
            with self.assertRaisesRegex(ValueError, 'cap'):
                bundle.verify(self.archive, max_expanded_bytes=1024)
        with self.assertRaisesRegex(ValueError, 'cap'):
            bundle.extract(self.archive, self.root / 'out', max_expanded_bytes=1024)
        self.assertFalse((self.root / 'out').exists())
        for invalid in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                bundle.verify(self.archive, max_expanded_bytes=invalid)
        with patch.object(bundle, 'MAX_INDEX_BYTES', 10):
            with self.assertRaisesRegex(ValueError, 'index exceeds'):
                bundle.verify(self.archive)

    def test_stream_limit_rejects_excess_and_failed_pack_removed(self):
        import io
        with self.assertRaisesRegex(ValueError, 'cap'):
            bundle._stream(io.BytesIO(b'x' * 30), limit=20)
        (self.source / 'data').write_bytes(b'x' * 2000)
        with self.assertRaisesRegex(ValueError, 'cap'):
            bundle.pack(self.source, self.archive, max_expanded_bytes=100)
        self.assertFalse(self.archive.exists())
        with self.assertRaisesRegex(ValueError, 'cap'):
            bundle.pack(self.source, self.archive, {'x': 'long replacement'}, max_expanded_bytes=2001)
        self.assertFalse(self.archive.exists())

    def test_outputs_must_be_fresh_and_bundle_outside_source(self):
        (self.source / 'data').write_text('x')
        bundle.pack(self.source, self.archive)
        before = self.archive.read_bytes()
        with self.assertRaises(FileExistsError):
            bundle.pack(self.source, self.archive)
        self.assertEqual(before, self.archive.read_bytes())
        for output in [self.source, self.source / 'data']:
            with self.assertRaises(FileExistsError):
                bundle.extract(self.archive, output)
        with self.assertRaisesRegex(ValueError, 'outside'):
            bundle.pack(self.source, self.source / 'inside.zip')

    def test_long_nested_filesystem_paths_pack_verify_extract_and_collision(self):
        native, base, source, member, data = self.long_path_fixture()
        archive = base / 'evidence.zip'
        output = base / 'replay/inputs'
        native(source / 'notes.yaml').write_bytes(b'native-host\n')
        packed = bundle.pack(source, archive, {'native-host': '[HOST]'})
        verified = bundle.verify(native(archive))
        extracted = bundle.extract(archive, output)
        self.assertEqual(verified['archive_sha256'], packed['archive_sha256'])
        self.assertEqual(extracted['bundle'], str(archive.absolute()))
        self.assertEqual(extracted['output'], str(output.absolute()))
        self.assertTrue(native(output).is_dir())
        self.assertEqual(native(output.joinpath(*member.split('/'))).read_bytes(), data)
        self.assertEqual(native(output / 'notes.yaml').read_bytes(), b'[HOST]\n')
        with zipfile.ZipFile(native(archive)) as contents:
            self.assertEqual(set(contents.namelist()), {member, 'notes.yaml', 'EXPORT.json'})
        with self.assertRaises(FileExistsError):
            bundle.extract(archive, output)
        with self.assertRaises(FileExistsError):
            bundle.pack(source, archive)

    def test_long_nested_paths_cleanup_failed_extract_and_pack(self):
        native, base, source, member, data = self.long_path_fixture()
        archive = base / 'evidence.zip'
        output = base / 'failed-replay/inputs'
        bundle.pack(source, archive)
        original_stream = bundle._stream

        def fail_write(source_stream, target=None, *, limit):
            if target is not None:
                target.write(b'partial')
                raise OSError('simulated write failure')
            return original_stream(source_stream, target, limit=limit)

        with patch.object(bundle, '_stream', side_effect=fail_write):
            with self.assertRaisesRegex(OSError, 'simulated write failure'):
                bundle.extract(archive, output)
        self.assertFalse(os.path.lexists(native(output)))
        failed_archive = base / 'failed-export.zip'
        with self.assertRaisesRegex(ValueError, 'cap'):
            bundle.pack(source, failed_archive, max_expanded_bytes=1)
        self.assertFalse(os.path.lexists(native(failed_archive)))

    def test_source_export_reserved_path_and_source_symlinks(self):
        (self.source / 'EXPORT.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'colli'):
            bundle.pack(self.source, self.archive)
        (self.source / 'EXPORT.json').unlink()
        (self.source / 'link').write_text('target')
        original_lstat = Path.lstat

        def linked_lstat(path):
            info = original_lstat(path)
            if path.name == 'link':
                fields = list(info)
                fields[stat.ST_MODE] = stat.S_IFLNK | 0o777
                return os.stat_result(fields)
            return info

        # Inject link metadata so this runs on Windows without symlink privilege.
        with patch.object(Path, 'lstat', linked_lstat):
            with self.assertRaisesRegex(ValueError, 'links'):
                bundle.pack(self.source, self.archive)

    def test_cli_pack_verify_extract_and_errors(self):
        (self.source / 'run.log').write_text('secret', encoding='utf-8')
        pairs = self.root / 'pairs.json'
        pairs.write_text(json.dumps({'secret': '[HOST]'}), encoding='utf-8')
        prefix = [sys.executable, '-B', str(TOOLS / 'evidence_bundle.py')]
        for arguments in [('pack', str(self.source), str(self.archive), '--replacements', str(pairs)),
                          ('verify', str(self.archive)),
                          ('extract', str(self.archive), str(self.root / 'out'))]:
            result = subprocess.run(prefix + list(arguments), capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['file_count'], 1)
        self.assertEqual((self.root / 'out/run.log').read_text(), '[HOST]')
        failed = subprocess.run(prefix + ['verify', str(self.archive), '--max-expanded-bytes', '1'], capture_output=True, text=True)
        self.assertEqual(failed.returncode, 1)
        self.assertIn('cap', failed.stderr)


if __name__ == '__main__':
    unittest.main()
