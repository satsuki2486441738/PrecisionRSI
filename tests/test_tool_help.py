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
                                 'compare_arrays', 'array_compare.py', 'compare_trees'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('# tree_compare\n'), 1)
        self.assertEqual(result.stdout.count('# array_compare\n'), 1)
        self.assertIn('compare_trees(', result.stdout)
        self.assertIn('leaf_adapter=None', result.stdout)
        self.assertIn('compare_arrays(', result.stdout)
        self.assertNotIn('def _flatten', result.stdout)
        self.assertNotIn('raise ValueError', result.stdout)

    def test_api_lookup_is_static_and_ambiguous_names_require_a_module(self):
        script = Path(__file__).resolve().parents[1]/'skills/precision-compare/scripts/tool_help.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root/'tool_help.py'
            entry.write_bytes(script.read_bytes())
            (root/'native.py').write_text('"""Native interface."""\n'
                'import dependency_that_is_not_installed\n'
                'raise RuntimeError("MUST_NOT_EXECUTE")\n'
                'def capture(value):\n    """Capture values."""\n'
                'async def collect(value): pass\n'
                'class Collector: pass\n')
            def run(*names):
                return subprocess.run([sys.executable, '-I', '-B', str(entry), *names],
                                      capture_output=True, text=True)
            for name in ('capture', 'collect', 'Collector'):
                result = run(name)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('# native\n', result.stdout)
                self.assertNotIn('MUST_NOT_EXECUTE', result.stdout)
            (root/'second.py').write_text('def capture(value): pass\n')
            result = run('native', 'capture')
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, '')
            self.assertIn('ambiguous API capture', result.stderr)
            self.assertIn('native, second', result.stderr)
            self.assertEqual(run('native').returncode, 0)
            (root/'capture.py').write_text('"""Preferred module."""\n')
            result = run('capture')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Preferred module.', result.stdout)

    def test_catalog_and_selected_apis_are_static_and_work_after_copy(self):
        script = Path(__file__).resolve().parents[1]/'skills/precision-compare/scripts/tool_help.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root/'tool_help.py'
            entry.write_bytes(script.read_bytes())
            (root/'native.py').write_text('''"""Capture native rows.

Inputs must share positions; returns scores.
"""
import dependency_that_is_not_installed
raise RuntimeError("MUST_NOT_EXECUTE")
def capture(value, *, dtype='float32'):
    """Collect one native row."""
    return "IMPLEMENTATION"
def unrelated(value):
    """UNREQUESTED_INTERFACE"""
async def collect(value): pass
class Collector:
    def capture(self, value): pass
''')
            (root/'second.py').write_text('"""Other collector."""\ndef capture(other): pass\n')
            (root/'_private.py').write_text('"""PRIVATE_MODULE"""\n')
            def run(*names):
                return subprocess.run([sys.executable, '-I', '-B', str(entry), *names],
                                      capture_output=True, text=True)
            result = run('--list')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, run().stdout)
            self.assertIn('| native | Capture native rows. |', result.stdout)
            self.assertIn('| second | Other collector. |', result.stdout)
            for omitted in ('Inputs must share', 'MUST_NOT_EXECUTE', 'PRIVATE_MODULE',
                            'UNREQUESTED_INTERFACE', 'IMPLEMENTATION'):
                self.assertNotIn(omitted, result.stdout)
            result = run('native:capture', 'native:collect', 'native:capture')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count('# native\n'), 1)
            self.assertEqual(result.stdout.count("capture(value, *, dtype='float32')"), 1)
            self.assertIn('collect(value)', result.stdout)
            self.assertIn('Inputs must share positions; returns scores.', result.stdout)
            self.assertNotIn('UNREQUESTED_INTERFACE', result.stdout)
            self.assertNotIn('class Collector', result.stdout)
            self.assertNotIn('IMPLEMENTATION', result.stdout)
            for names in [('native:capture', 'native'), ('native', 'native:capture')]:
                self.assertIn('UNREQUESTED_INTERFACE', run(*names).stdout)
            self.assertIn('Collector.capture(self, value)', run('Collector').stdout)
            self.assertNotIn('UNREQUESTED_INTERFACE', run('Collector').stdout)
            result = run('collect')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('collect(value)', result.stdout)
            self.assertNotIn('UNREQUESTED_INTERFACE', result.stdout)
            for names in [('--list', 'native'), ('native:missing',), ('missing:capture',),
                          ('native:_private',), ('native:capture:extra',)]:
                result = run(*names)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, '')

    def test_declarations_default_and_full_contracts_are_distinct_and_static(self):
        script = Path(__file__).resolve().parents[1]/'skills/precision-compare/scripts/tool_help.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root/'tool_help.py'
            entry.write_bytes(script.read_bytes())
            (root/'native.py').write_text('''"""Capture selected rows.
Call: capture(values)
Input: finite rows at matching positions.
Output: scores and coverage.
Requires: native runtime; preserve dtype.

Details:
SCHEMA_DETAIL: rows=[{position:int, values:array}].
"""
import dependency_that_is_not_installed
raise RuntimeError("MUST_NOT_EXECUTE")
def capture(values, *, dtype='float32'):
    """Capture matching observations.

Details:
METHOD_DETAIL: native dtype must be recorded.
"""
    return "IMPLEMENTATION"
def helper(value):
    """SIBLING_INTERFACE"""
''')
            def run(*names):
                return subprocess.run([sys.executable, '-I', '-B', str(entry), *names],
                                      capture_output=True, text=True)
            short = run('native')
            self.assertEqual(short.returncode, 0, short.stderr)
            self.assertIn('Input: finite rows at matching positions.', short.stdout)
            self.assertNotIn('SCHEMA_DETAIL', short.stdout)
            self.assertNotIn('SIBLING_INTERFACE', short.stdout)
            selected = run('capture')
            self.assertEqual(selected.returncode, 0, selected.stderr)
            self.assertIn("capture(values, *, dtype='float32')", selected.stdout)
            self.assertIn('Capture matching observations.', selected.stdout)
            self.assertNotIn('SCHEMA_DETAIL', selected.stdout)
            self.assertNotIn('METHOD_DETAIL', selected.stdout)
            expanded = run('--full', 'capture')
            self.assertEqual(expanded.returncode, 0, expanded.stderr)
            self.assertIn('SCHEMA_DETAIL', expanded.stdout)
            self.assertIn('METHOD_DETAIL', expanded.stdout)
            self.assertNotIn('SIBLING_INTERFACE', expanded.stdout)
            batch = run('--full', 'native', 'capture', 'native')
            self.assertEqual(batch.returncode, 0, batch.stderr)
            self.assertEqual(batch.stdout.count('# native\n'), 1)
            self.assertIn('SIBLING_INTERFACE', batch.stdout)
            self.assertIn('SCHEMA_DETAIL', batch.stdout)
            self.assertGreater(len(batch.stdout), len(short.stdout))
            for text in (short.stdout, selected.stdout, expanded.stdout, batch.stdout):
                self.assertNotIn('MUST_NOT_EXECUTE', text)
                self.assertNotIn('IMPLEMENTATION', text)
            for names in [('--full',), ('--list', '--full'), ('--full', 'native', 'missing')]:
                result = run(*names)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, '')

    def test_direct_cli_help_retains_flags_without_expanding_contracts(self):
        scripts = Path(__file__).resolve().parents[1]/'skills/precision-compare/scripts'
        result = subprocess.run([sys.executable, '-I', '-S', '-B', str(scripts/'record_run.py'),
                                 '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--timeout', result.stdout)
        self.assertIn('--tail-bytes', result.stdout)
        self.assertIn('Input:', result.stdout)
        self.assertIn('Output:', result.stdout)
        self.assertNotIn('Details:', result.stdout)
        full = subprocess.run([sys.executable, '-I', '-S', '-B', str(scripts/'tool_help.py'),
                               '--full', 'record_run'], capture_output=True, text=True)
        self.assertEqual(full.returncode, 0, full.stderr)
        self.assertIn('Details:', full.stdout)

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
