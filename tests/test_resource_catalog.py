"""Verified-resource discovery must avoid stale reuse and preserve registration history."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/precision-compare/scripts'))
from resource_catalog import find, register


@unittest.skipUnless(os.name == 'posix', 'POSIX catalog lock')
class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = self.root / 'shared/catalog.json'
        self.entry = self.root / 'entry.json'
        (self.root / 'adapter.py').write_text('version = 1\n')
        (self.root / 'runtime.json').write_text('{"python":"native"}')
        (self.root / 'command.json').write_text(json.dumps(dict(status='completed', returncode=0, argv=['native', 'check'])))
        self.spec = dict(id='torch-cpu', kind='environment', properties=dict(framework='torch', device='cpu'),
                         paths=dict(adapter='adapter.py', runtime='runtime.json'), evidence='command.json')
        self.entry.write_text(json.dumps(self.spec))

    def test_registration_discovery_and_idempotence_across_tasks(self):
        first = register(self.catalog, self.entry)
        self.assertTrue(first['registered'])
        second = register(self.catalog, self.entry)
        self.assertFalse(second['registered'])
        found = find(self.catalog, 'environment', dict(device='cpu'))
        self.assertEqual(len(found), 1)
        self.assertEqual(Path(found[0]['paths']['adapter']['path']).read_text(), 'version = 1\n')
        self.assertEqual(find(self.catalog, 'adapter'), [])
        self.assertEqual(find(self.catalog, properties=dict(device='npu')), [])

    def test_missing_and_changed_resources_are_not_returned(self):
        register(self.catalog, self.entry)
        (self.root / 'adapter.py').write_text('version = 2\n')
        self.assertEqual(find(self.catalog), [])
        register(self.catalog, self.entry)
        self.assertEqual(len(find(self.catalog)), 1)
        (self.root / 'runtime.json').unlink()
        self.assertEqual(find(self.catalog), [])
        self.assertEqual(len(json.loads(self.catalog.read_text())['versions']), 2)

    def test_evidence_failure_cannot_register_or_validate_a_resource(self):
        evidence = self.root / 'command.json'
        evidence.write_text(json.dumps(dict(status='failed', returncode=1, argv=['native'])))
        with self.assertRaisesRegex(ValueError, 'successful native'):
            register(self.catalog, self.entry)
        self.assertFalse(self.catalog.exists())
        evidence.write_text(json.dumps(dict(status='completed', returncode=0, argv=['native'])))
        register(self.catalog, self.entry)
        evidence.write_text(json.dumps(dict(status='failed', returncode=1, argv=['native'])))
        self.assertEqual(find(self.catalog), [])

    def test_properties_keep_types_and_latest_registration_owns_scope(self):
        self.spec['properties']['tp'] = 1
        self.entry.write_text(json.dumps(self.spec))
        register(self.catalog, self.entry)
        self.assertEqual(find(self.catalog, properties=dict(tp=True)), [])
        self.assertEqual(len(find(self.catalog, properties=dict(tp=1))), 1)
        self.spec['properties']['device'] = 'npu'
        self.entry.write_text(json.dumps(self.spec))
        register(self.catalog, self.entry)
        self.assertEqual(find(self.catalog, properties=dict(device='cpu')), [])
        self.assertEqual(len(find(self.catalog, properties=dict(device='npu'))), 1)

    def test_discovery_of_absent_catalog_is_read_only(self):
        self.assertEqual(find(self.catalog), [])
        self.assertFalse(self.catalog.parent.exists())

    def test_directory_and_symlink_identity_invalidation(self):
        folder = self.root / 'environment'
        folder.mkdir()
        link = folder / 'source'
        link.symlink_to(self.root / 'adapter.py')
        self.spec['paths']['directory'] = 'environment'
        self.entry.write_text(json.dumps(self.spec))
        register(self.catalog, self.entry)
        self.assertEqual(len(find(self.catalog)), 1)
        (self.root / 'adapter.py').write_text('changed')
        self.assertEqual(find(self.catalog), [])


if __name__ == '__main__':
    unittest.main()
