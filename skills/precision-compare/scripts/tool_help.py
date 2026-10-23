#!/usr/bin/env python3
"""Discover tool interfaces without imports or module execution (standard library).

CLI: python scripts/tool_help.py [--list | TOOL [TOOL ...]]
No arguments or --list: compact module/purpose catalog.
TOOL: module name, unique public API name, or module:API for ambiguous names.
Module names take precedence. API queries return the owning module's contract
and only the selected declarations; module queries return all public interfaces.
Batch queries share each module contract once. Source bodies/private helpers are
omitted. Contracts, signatures and catalog summaries come from the current AST;
no separate registry or execution-framework dependency is required.
"""
import argparse
import ast
from pathlib import Path


def _parse(path):
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _public(tree):
    return {node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and not node.name.startswith('_') and node.name != 'main'}


def catalog(directory):
    """List module purposes from current docstrings without dependency imports."""
    lines = ['| Module | Purpose |', '|---|---|']
    for path in sorted(Path(directory).glob('*.py')):
        if path.stem.startswith('_'):
            continue
        doc = ast.get_docstring(_parse(path)) or ''
        summary = doc.splitlines()[0] if doc else ''
        lines.append('| ' + path.stem + ' | ' + summary.replace('|', r'\|') + ' |')
    return '\n'.join(lines) + '\n'


def describe(path, selected=None):
    """Extract a module contract and all or selected public declarations statically."""
    path = Path(path)
    tree = _parse(path)
    declarations = _public(tree)
    if selected is not None:
        missing = set(selected) - declarations.keys()
        if missing:
            raise ValueError('unknown public API: ' + ', '.join(sorted(missing)))
    lines = ['# ' + path.stem, ast.get_docstring(tree) or '']

    def function(node, prefix=''):
        signature = prefix + node.name + '(' + ast.unparse(node.args) + ')'
        if node.returns is not None:
            signature += ' -> ' + ast.unparse(node.returns)
        lines.extend(['\n' + signature, ast.get_docstring(node) or ''])

    for name, node in declarations.items():
        if selected is not None and name not in selected:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function(node)
        else:
            lines.extend(['\nclass ' + node.name, ast.get_docstring(node) or ''])
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not method.name.startswith('_') or method.name == '__init__':
                        function(method, node.name + '.')
    return '\n'.join(lines).rstrip() + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list', action='store_true', help='list module purposes only')
    parser.add_argument('tools', nargs='*', metavar='TOOL')
    args = parser.parse_args(argv)
    if args.list and args.tools:
        parser.error('use --list without tool names')
    directory = Path(__file__).parent
    requests, exports = {}, None
    try:
        if not args.tools:
            print(catalog(directory), end='')
            return 0
        for raw in args.tools:
            owner, separator, api = raw.partition(':')
            name = owner.removesuffix('.py')
            if (not name.isidentifier() or name.startswith('_') or
                    (separator and (not api.isidentifier() or api.startswith('_')))):
                parser.error('expected a module, API name or module:API: ' + raw)
            path = directory / (name + '.py')
            selected = None
            if separator:
                if not path.is_file() or api not in _public(_parse(path)):
                    parser.error('unknown public API: ' + raw)
                selected = [api]
            elif not path.is_file():
                if exports is None:
                    exports = {}
                    for module in sorted(directory.glob('*.py')):
                        if not module.stem.startswith('_'):
                            for symbol in _public(_parse(module)):
                                exports.setdefault(symbol, []).append(module)
                matches = exports.get(name, [])
                if not matches:
                    parser.error('unknown tool: ' + raw)
                if len(matches) != 1:
                    parser.error('ambiguous API ' + name + '; use module:API: ' +
                                 ', '.join(p.stem for p in matches))
                path, selected = matches[0], [name]
            if path not in requests or selected is None:
                requests[path] = selected
            elif requests[path] is not None:
                requests[path] = list(dict.fromkeys(requests[path] + selected))
        text = '\n'.join(describe(path, selected) for path, selected in requests.items())
    except (OSError, SyntaxError, ValueError) as exc:
        parser.error(str(exc))
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
