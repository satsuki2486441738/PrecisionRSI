"""Interface discovery reads source declarations without dependency imports."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'skills/precision-compare/scripts'))
from tool_help import describe


class ToolHelpTests(unittest.TestCase):
    def test_interfaces_are_extracted_without_imports_or_top_level_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'native.py'
            path.write_text('''"""Native tool interface."""
import dependency_that_is_not_installed
raise RuntimeError("MUST_NOT_EXECUTE")
def compare(a, /, b=2, *, metrics=False) -> dict:
    """Compare the requested observations."""
    return "IMPLEMENTATION_BODY"
def _private():
    """PRIVATE_DETAIL"""
class Collector:
    """Native boundary collector."""
    def __init__(self, config): pass
    def capture(self, value):
        """Capture selected values."""
        return "CLASS_BODY"
    def _internal(self): pass
''')
            text = describe(path)
        self.assertIn('Native tool interface.', text)
        self.assertIn('compare(a, /, b=2, *, metrics=False) -> dict', text)
        self.assertIn('Collector.__init__(self, config)', text)
        self.assertIn('Collector.capture(self, value)', text)
        self.assertIn('Capture selected values.', text)
        for omitted in ('dependency_that_is_not_installed', 'MUST_NOT_EXECUTE',
                        'IMPLEMENTATION_BODY', 'PRIVATE_DETAIL', 'CLASS_BODY', '_internal'):
            self.assertNotIn(omitted, text)

    def test_cli_batches_deduplicates_and_supports_isolated_python(self):
        script = Path(__file__).resolve().parents[1]/'skills/precision-compare/scripts/tool_help.py'
        result = subprocess.run([sys.executable, '-I', '-B', str(script), 'tree_compare',
                                 'array_compare.py', 'tree_compare'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('# tree_compare\n'), 1)
        self.assertEqual(result.stdout.count('# array_compare\n'), 1)
        self.assertIn('compare_trees(', result.stdout)
        self.assertIn('leaf_adapter=None', result.stdout)
        self.assertIn('compare_arrays(', result.stdout)
        self.assertNotIn('def _flatten', result.stdout)
        self.assertNotIn('raise ValueError', result.stdout)

    def test_cli_rejects_unknown_names_and_paths_without_partial_output(self):
        script = Path(__file__).resolve().parents[1]/'skills/precision-compare/scripts/tool_help.py'
        for name in ('not_a_tool', '../array_compare', '/tmp/array_compare', '_private'):
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, '-I', '-B', str(script),
                                         'array_compare', name], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, '')
                self.assertIn('error:', result.stderr)


if __name__ == '__main__':
    unittest.main()
