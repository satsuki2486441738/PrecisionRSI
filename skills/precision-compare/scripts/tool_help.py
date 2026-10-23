#!/usr/bin/env python3
"""Show tool interfaces without importing frameworks or executing tool modules.

CLI: python scripts/tool_help.py TOOL [TOOL ...]
Use installed tool names such as array_compare, group_metrics or tree_compare.
The output contains module/API docstrings and public function/class signatures
read from the current source. Implementations and private helpers are omitted.
"""
import argparse
import ast
from pathlib import Path


def describe(path):
    """Extract interface text from a Python source file without executing it."""
    path = Path(path)
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    lines = ['# ' + path.stem, ast.get_docstring(tree) or '']

    def function(node, prefix=''):
        signature = prefix + node.name + '(' + ast.unparse(node.args) + ')'
        if node.returns is not None:
            signature += ' -> ' + ast.unparse(node.returns)
        lines.extend(['\n' + signature, ast.get_docstring(node) or ''])

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith('_') and node.name != 'main':
                function(node)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
            lines.extend(['\nclass ' + node.name, ast.get_docstring(node) or ''])
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not method.name.startswith('_') or method.name == '__init__':
                        function(method, node.name + '.')
    return '\n'.join(lines).rstrip() + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('tools', nargs='+', metavar='TOOL')
    args = parser.parse_args(argv)
    paths = []
    for raw in args.tools:
        name = raw.removesuffix('.py')
        if not name.isidentifier() or name.startswith('_'):
            parser.error('expected an installed tool name: ' + raw)
        path = Path(__file__).with_name(name + '.py')
        if not path.is_file():
            parser.error('unknown tool: ' + raw)
        if path not in paths:
            paths.append(path)
    try:
        text = '\n'.join(describe(path) for path in paths)
    except (OSError, SyntaxError) as exc:
        parser.error(str(exc))
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
