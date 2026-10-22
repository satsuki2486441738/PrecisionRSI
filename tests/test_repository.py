"""Regressions for repository metadata and Markdown maintenance checks."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('repository_validation', Path(__file__).resolve().parents[1] / '.github/validate.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


smoke_spec = importlib.util.spec_from_file_location(
    'smoke_install', Path(__file__).resolve().parents[1] / '.github/smoke_install.py')
smoke = importlib.util.module_from_spec(smoke_spec)
smoke_spec.loader.exec_module(smoke)


class RepositoryValidationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_links_references_images_unicode_duplicates_and_code(self):
        target = self.root / '有 空格.md'
        target.write_text('# 标题\n# Repeat\n# Repeat\n# Repeat-1\n', encoding='utf-8')
        source = self.root / 'README.md'
        source.write_text('''[标题](有%20空格.md#标题)
[repeat][target]

[target]: 有%20空格.md#repeat-1

![image](有%20空格.md)
[duplicate](有%20空格.md#repeat-1-1)
`[inline](missing.md)`
```
[fenced](missing.md)
```
[external](https://example.com/missing)
''', encoding='utf-8')
        count, errors = validator.validate_links(self.root, [source])
        self.assertEqual((count, errors), (4, []))
        source.write_text('[bad](missing.md)\n[bad](有%20空格.md#absent)', encoding='utf-8')
        count, errors = validator.validate_links(self.root, [source])
        self.assertEqual(count, 2)
        self.assertEqual(len(errors), 2)
        self.assertIn('missing local target', errors[0])
        self.assertIn('missing heading', errors[1])

    def test_all_skill_packages_are_validated(self):
        skills = self.root / 'skills'
        skills.mkdir()
        with self.assertRaisesRegex(ValueError, 'no installable skills'):
            validator.validate_skills(self.root)
        for name in ('precision-compare', 'precision-repair'):
            folder = skills / name
            folder.mkdir()
            (folder / 'SKILL.md').write_text(
                f'---\nname: {name}\ndescription: Execute the task.\n---\n# Task\n')
        self.assertEqual([path.name for path in validator.validate_skills(self.root)],
                         ['precision-compare', 'precision-repair'])
        entrypoint = skills / 'precision-repair/SKILL.md'
        entrypoint.write_text('---\nname: wrong-name\ndescription: Repair.\n---\n')
        with self.assertRaisesRegex(ValueError, 'directory mismatch'):
            validator.validate_skills(self.root)
        entrypoint.unlink()
        with self.assertRaisesRegex(ValueError, 'missing SKILL.md'):
            validator.validate_skills(self.root)

    def test_clean_install_preserves_companion_links_and_omits_caches(self):
        source = self.root / 'checkout/skills'
        for name in ('precision-compare', 'precision-repair'):
            folder = source / name
            folder.mkdir(parents=True)
            (folder / 'SKILL.md').write_text(
                f'---\nname: {name}\ndescription: Execute the task.\n---\n# Task\n')
        repair = source / 'precision-repair/SKILL.md'
        with repair.open('a') as stream:
            stream.write('[tools](../precision-compare/scripts/tool.py)\n')
        scripts = source / 'precision-compare/scripts'
        scripts.mkdir()
        (scripts / 'tool.py').write_text('print("tool")\n')
        (scripts / '__pycache__').mkdir()
        (scripts / '__pycache__/tool.pyc').write_bytes(b'cache')
        (source.parent / 'private.txt').write_text('case-specific')
        installed = self.root / 'installed'
        smoke.install_skills(source, installed / 'skills')
        self.assertEqual(len(validator.validate_skills(installed)), 2)
        self.assertEqual(validator.validate_links(installed, [installed / 'skills/precision-repair/SKILL.md']),
                         (1, []))
        self.assertFalse((installed / 'private.txt').exists())
        self.assertFalse((installed / 'skills/precision-compare/scripts/__pycache__').exists())
        self.assertEqual((installed / 'skills/precision-compare/scripts/tool.py').read_text(),
                         'print("tool")\n')

    def test_published_collection_installs_linked_skills(self):
        source = Path(__file__).resolve().parents[1] / 'skills'
        installed = self.root / 'clean-install'
        smoke.install_skills(source, installed / 'skills')
        folders = validator.validate_skills(installed)
        self.assertEqual([p.name for p in folders],
                         ['framework-adapt', 'precision-compare', 'precision-repair'])
        docs = sorted((installed / 'skills').rglob('*.md'))
        _, errors = validator.validate_links(installed, docs)
        self.assertEqual(errors, [])
        text = (installed / 'skills/precision-repair/SKILL.md').read_text()
        self.assertIn('../precision-compare/references/comparison.md', text)
        self.assertTrue((installed / 'skills/precision-compare/scripts/replay_bundle.py').is_file())

    def test_smoke_artifact_directory_accepts_symlink_spelling(self):
        selected = self.root / 'selected'
        selected.mkdir()
        alias = self.root / 'alias'
        alias.symlink_to(selected, target_is_directory=True)
        artifacts = selected / 'run'
        artifacts.mkdir()
        for reported, output_root in ((artifacts, alias), (alias / 'run', selected)):
            with self.subTest(reported=reported, output_root=output_root):
                accepted = smoke.validate_artifact_directory(reported, output_root)
                self.assertTrue(accepted.samefile(artifacts))

    def test_smoke_artifact_directory_rejects_different_directory(self):
        selected = self.root / 'selected'
        selected.mkdir()
        other = self.root / 'other'
        other.mkdir()
        artifacts = other / 'run'
        artifacts.mkdir()
        escaped = selected / 'linked-run'
        escaped.symlink_to(artifacts, target_is_directory=True)
        for reported in (artifacts, escaped):
            with self.subTest(reported=reported):
                with self.assertRaisesRegex(ValueError, 'outside selected output root'):
                    smoke.validate_artifact_directory(reported, selected)

    def test_metadata_name_description_optional_fields(self):
        folder = self.root / 'test-skill'
        folder.mkdir()
        path = folder / 'SKILL.md'
        valid = 'name: test-skill\ndescription: Run a comparison.\ncompatibility: Python 3.10+\nmetadata:\n  author: example\n'
        path.write_text('---\n' + valid + '---\n# Skill\n')
        validator.validate_metadata(path)
        for content in ('# Missing frontmatter', '---\n[]\n---\n',
                        '---\n' + valid.replace('test-skill', 'other') + '---\n',
                        '---\n' + valid.replace('Run a comparison.', '123') + '---\n',
                        '---\n' + valid + 'unknown: x\n---\n'):
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    validator.validate_metadata(path)


if __name__ == '__main__':
    unittest.main()
