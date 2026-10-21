"""Regressions for repository metadata and Markdown maintenance checks."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('repository_validation', Path(__file__).resolve().parents[1] / '.github/validate.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


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
